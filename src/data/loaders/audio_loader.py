"""Audio DataLoader utilities tailored for CTC speech recognition."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocessing.audio_preprocessor import AudioPreprocessor, AudioPreprocessConfig
from src.data.preprocessing.text_tokenizer import TextTokenizer, DEFAULT_VOCAB


@dataclass(slots=True)
class AudioSample:
    path: Path
    transcript: str
    split: str = "train"


class AudioCTCDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[AudioSample],
        preprocessor: AudioPreprocessor,
        tokenizer: TextTokenizer,
        augment: bool = False,
    ) -> None:
        self.entries = list(entries)
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        features, feature_length = self.preprocessor.process_file(entry.path, augment=self.augment)
        tokens = torch.tensor(self.tokenizer.encode(entry.transcript), dtype=torch.long)
        return {
            "features": features,
            "feature_length": feature_length,
            "targets": tokens,
            "target_length": tokens.size(0),
            "transcript": entry.transcript,
        }


def ctc_collate_fn(batch: List[dict]) -> dict:
    batch_size = len(batch)
    feature_lengths = torch.tensor([item["feature_length"] for item in batch], dtype=torch.long)
    max_len = int(feature_lengths.max())
    feature_dim = batch[0]["features"].size(1)
    inputs = torch.zeros(batch_size, max_len, feature_dim)

    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    total_target_len = int(target_lengths.sum())
    targets = torch.zeros(total_target_len, dtype=torch.long)

    t_offset = 0
    for idx, item in enumerate(batch):
        length = item["feature_length"]
        inputs[idx, :length] = item["features"]
        target_len = item["target_length"]
        targets[t_offset : t_offset + target_len] = item["targets"]
        t_offset += target_len

    return {
        "inputs": inputs,
        "input_lengths": feature_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "transcripts": [item["transcript"] for item in batch],
    }


class AudioDataModule:
    """Convenience wrapper that builds train/val/test DataLoaders."""

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_root: str | Path | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        preprocess_config: AudioPreprocessConfig | None = None,
        vocab: List[str] | None = None,
        seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.audio_root = Path(audio_root) if audio_root else self.metadata_csv.parent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.preprocessor = AudioPreprocessor(preprocess_config)
        vocab = vocab or DEFAULT_VOCAB
        self.tokenizer = TextTokenizer(vocab=vocab)

        self.samples = self._load_metadata()
        if len(self.samples) < 100:
            raise ValueError("Se requieren al menos 100 audios etiquetados para entrenar el modelo CTC.")

        self._ensure_splits()

    def _load_metadata(self) -> List[AudioSample]:
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"No se encontró metadata: {self.metadata_csv}")
        with self.metadata_csv.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            required = {"path", "text"}
            if not required.issubset(reader.fieldnames or {}):
                raise ValueError(f"El CSV debe contener columnas {required}")
            samples: List[AudioSample] = []
            for row in reader:
                rel_path = row["path"].strip()
                transcript = row["text"].strip()
                split = row.get("split", "train").strip().lower() or "train"
                audio_path = self.audio_root / rel_path
                samples.append(AudioSample(path=audio_path, transcript=transcript, split=split))
        return samples

    def _ensure_splits(self) -> None:
        train = [s for s in self.samples if s.split == "train"]
        val = [s for s in self.samples if s.split == "val"]
        test = [s for s in self.samples if s.split == "test"]

        rng = random.Random(self.seed)
        if not val:
            rng.shuffle(train)
            split_idx = max(1, int(len(train) * self.val_ratio))
            val.extend(train[:split_idx])
            for sample in val:
                sample.split = "val"
            train = train[split_idx:]
        if not test and train:
            rng.shuffle(train)
            split_idx = max(1, int(len(train) * self.test_ratio))
            test.extend(train[:split_idx])
            for sample in test:
                sample.split = "test"
            train = train[split_idx:]

        if not train:
            raise ValueError("No hay muestras para entrenamiento tras crear splits.")

        self.samples = train + val + test

    def _build_dataset(self, split: str) -> AudioCTCDataset:
        entries = [s for s in self.samples if s.split == split]
        augment = split == "train"
        return AudioCTCDataset(entries, self.preprocessor, self.tokenizer, augment=augment)

    def dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        train_ds = self._build_dataset("train")
        val_ds = self._build_dataset("val")
        test_entries = [s for s in self.samples if s.split == "test"]
        test_ds = AudioCTCDataset(test_entries, self.preprocessor, self.tokenizer, augment=False) if test_entries else None

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ctc_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ctc_collate_fn,
        )
        test_loader = (
            DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=ctc_collate_fn,
            )
            if test_ds
            else None
        )
        return train_loader, val_loader, test_loader

    def export_metadata(self) -> dict:
        return {
            "tokenizer": self.tokenizer.to_dict(),
            "preprocess": self.preprocessor.to_dict(),
            "sample_rate": self.preprocessor.config.sample_rate,
            "feature_dim": self.preprocessor.feature_dim,
        }
