"""Lightweight tokenizer for character-level CTC transcriptions."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List


DEFAULT_VOCAB = list("abcdefghijklmnopqrstuvwxyzáéíóúüñ0123456789 '?.,!")


@dataclass(slots=True)
class TextTokenizer:
    vocab: List[str]
    blank_symbol: str = "<blank>"
    symbols: List[str] | None = None
    char2idx: Dict[str, int] | None = None
    idx2char: Dict[int, str] | None = None

    def __post_init__(self) -> None:
        if self.blank_symbol in self.vocab:
            raise ValueError("blank_symbol must not overlap with vocab")
        self.vocab = self.vocab.copy()
        self.symbols = [self.blank_symbol] + self.vocab
        self.char2idx = {ch: idx + 1 for idx, ch in enumerate(self.vocab)}
        self.char2idx[self.blank_symbol] = 0
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}

    @property
    def blank_idx(self) -> int:
        return 0

    def normalize_text(self, text: str) -> str:
        text = text.strip().lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return text

    def encode(self, text: str) -> List[int]:
        normalized = self.normalize_text(text)
        tokens = []
        for ch in normalized:
            if ch in self.char2idx:
                tokens.append(self.char2idx[ch])
        if not tokens:
            raise ValueError(f"Texto vacío tras normalización: '{text}'")
        return tokens

    def decode(self, indices: List[int], collapse_repeats: bool = True) -> str:
        decoded = []
        prev = None
        for idx in indices:
            if idx == self.blank_idx:
                prev = None
                continue
            if collapse_repeats and idx == prev:
                continue
            decoded.append(self.idx2char.get(idx, ""))
            prev = idx
        return "".join(decoded).replace("  ", " ").strip()

    def to_dict(self) -> dict:
        return {
            "vocab": self.vocab,
            "blank_symbol": self.blank_symbol,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TextTokenizer":
        return cls(vocab=data["vocab"], blank_symbol=data.get("blank_symbol", "<blank>"))
