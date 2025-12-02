"""Audio preprocessing utilities for speech-to-text training."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import torch
import torchaudio


@dataclass(slots=True)
class AudioPreprocessConfig:
    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    f_min: int = 0
    f_max: int = 8_000
    use_mfcc: bool = True
    n_mfcc: int = 40
    spec_augment: bool = True
    time_mask_param: int = 30
    freq_mask_param: int = 15


class AudioPreprocessor:
    """Turn raw waveforms into log-mel (and optional MFCC) features."""

    def __init__(self, config: AudioPreprocessConfig | None = None):
        self.config = config or AudioPreprocessConfig()
        cfg = self.config

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.n_mels,
            power=2.0,
            normalized=True,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.mfcc_transform = (
            torchaudio.transforms.MFCC(
                sample_rate=cfg.sample_rate,
                n_mfcc=cfg.n_mfcc,
                melkwargs={
                    "n_mels": cfg.n_mels,
                    "n_fft": cfg.n_fft,
                    "hop_length": cfg.hop_length,
                    "f_min": cfg.f_min,
                    "f_max": cfg.f_max,
                },
            )
            if cfg.use_mfcc
            else None
        )

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=cfg.time_mask_param)

    @property
    def feature_dim(self) -> int:
        base = self.config.n_mels
        if self.config.use_mfcc:
            base += self.config.n_mfcc
        return base

    def load_waveform(self, path: str | Path) -> torch.Tensor:
        """Load audio file, convert to mono, and resample."""
        waveform, sr = torchaudio.load(str(path))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.config.sample_rate)
        return waveform

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean()
        std = features.std().clamp(min=1e-5)
        return (features - mean) / std

    def _stack_features(self, mel: torch.Tensor, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.db_transform(mel)
        if self.mfcc_transform is None:
            return mel
        mfcc = self.mfcc_transform(waveform)
        return torch.cat([mel, mfcc], dim=1)

    def extract_features(self, waveform: torch.Tensor, apply_augment: bool = False) -> Tuple[torch.Tensor, int]:
        """Return time-major features and their valid length."""
        mel = self.mel_transform(waveform)
        stacked = self._stack_features(mel, waveform)
        if apply_augment and self.config.spec_augment:
            stacked = self.freq_mask(stacked)
            stacked = self.time_mask(stacked)
        stacked = self._normalize(stacked)
        stacked = stacked.squeeze(0).transpose(0, 1).contiguous()  # (time, dims)
        length = stacked.size(0)
        return stacked, length

    def process_file(self, path: str | Path, augment: bool = False) -> Tuple[torch.Tensor, int]:
        waveform = self.load_waveform(path)
        return self.extract_features(waveform, apply_augment=augment)

    def to_dict(self) -> dict:
        return asdict(self.config)
