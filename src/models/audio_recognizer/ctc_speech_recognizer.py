"""CTC-based speech recognizer built with PyTorch."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.base_model import BaseClassifier
from src.data.preprocessing.text_tokenizer import TextTokenizer


def _wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return float(len(hyp_words) > 0)
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / max(1, len(ref_words))


class _CTCNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 100, feature_dim)
            conv_out = self.conv(dummy)
            conv_dim = conv_out.size(1) * conv_out.size(3)
        self.rnn = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # features: (B, T, F)
        x = features.unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)
        bsz, channels, timesteps, freq = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(bsz, timesteps, channels * freq)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)


@dataclass(slots=True)
class CTCSpeechConfig:
    vocab: List[str]
    feature_dim: int
    blank_symbol: str = "<blank>"
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    lr: float = 1e-3
    grad_clip: float = 5.0


class CTCSpeechRecognizer(BaseClassifier):
    def __init__(self, config: Dict):
        super().__init__(config.get("device"))
        self.config = CTCSpeechConfig(
            vocab=config["vocab"],
            feature_dim=config["feature_dim"],
            blank_symbol=config.get("blank_symbol", "<blank>"),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.2),
            lr=config.get("lr", 1e-3),
            grad_clip=config.get("grad_clip", 5.0),
        )
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = TextTokenizer(self.config.vocab, blank_symbol=self.config.blank_symbol)
        self.num_classes = len(self.config.vocab) + 1
        self.ctc_loss = nn.CTCLoss(blank=self.tokenizer.blank_idx, zero_infinity=True)
        self.history: Dict[str, List[float]] | None = None
        self.model: _CTCNetwork | None = None

    def build_model(self, num_classes: int | None = None) -> None:  # type: ignore[override]
        self.model = _CTCNetwork(
            feature_dim=self.config.feature_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            num_classes=num_classes or self.num_classes,
        ).to(self.device)

    def _ensure_model(self) -> _CTCNetwork:
        if self.model is None:
            self.build_model()
        assert self.model is not None
        return self.model

    def _move_batch(self, batch: dict) -> dict:
        return {
            "inputs": batch["inputs"].to(self.device),
            "input_lengths": batch["input_lengths"].to(self.device),
            "targets": batch["targets"].to(self.device),
            "target_lengths": batch["target_lengths"].to(self.device),
            "transcripts": batch["transcripts"],
        }

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 30,
        lr: float | None = None,
    ) -> Dict[str, List[float]]:
        model = self._ensure_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr or self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        history = {"train_loss": [], "val_loss": [], "val_wer": []}

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = self._move_batch(batch)
                optimizer.zero_grad()
                log_probs = model(batch["inputs"], batch["input_lengths"])
                log_probs_t = log_probs.transpose(0, 1)
                loss = self.ctc_loss(log_probs_t, batch["targets"], batch["input_lengths"], batch["target_lengths"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_train_loss)

            if val_loader is not None:
                val_loss, val_wer = self.evaluate(val_loader, compute_loss=True)
                history["val_loss"].append(val_loss)
                history["val_wer"].append(val_wer)
                scheduler.step(val_loss)
            else:
                history["val_loss"].append(0.0)
                history["val_wer"].append(0.0)

        self.history = history
        return history

    def _greedy_decode(self, log_probs: torch.Tensor, lengths: Sequence[int]) -> List[str]:
        decoded = []
        preds = log_probs.argmax(dim=-1).cpu()
        for idx, length in enumerate(lengths):
            token_ids = preds[idx, :length].tolist()
            decoded.append(self.tokenizer.decode(token_ids))
        return decoded

    def evaluate(self, data_loader, compute_loss: bool = False) -> tuple[float, float]:
        model = self._ensure_model()
        model.eval()
        total_loss = 0.0
        total_wer = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch(batch)
                log_probs = model(batch["inputs"], batch["input_lengths"])
                if compute_loss:
                    log_probs_t = log_probs.transpose(0, 1)
                    loss = self.ctc_loss(log_probs_t, batch["targets"], batch["input_lengths"], batch["target_lengths"])
                    total_loss += loss.item()
                predictions = self._greedy_decode(log_probs, batch["input_lengths"].tolist())
                for pred, target in zip(predictions, batch["transcripts"]):
                    total_wer += _wer(target, pred)
                total_samples += len(predictions)

        avg_loss = total_loss / max(1, len(data_loader)) if compute_loss else 0.0
        avg_wer = total_wer / max(1, total_samples)
        return avg_loss, avg_wer

    def transcribe(self, features: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        model = self._ensure_model()
        model.eval()
        with torch.no_grad():
            log_probs = model(features.to(self.device), lengths.to(self.device))
        return self._greedy_decode(log_probs, lengths.tolist())

    def save_model(self, path: str) -> None:
        model = self._ensure_model()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(self.config),
                "tokenizer": self.tokenizer.to_dict(),
            },
            path,
        )

    def load_model(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.config = CTCSpeechConfig(**checkpoint["config"])
        self.tokenizer = TextTokenizer.from_dict(checkpoint["tokenizer"])
        self.num_classes = len(self.tokenizer.vocab) + 1
        self.build_model()
        assert self.model is not None
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
