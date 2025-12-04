"""Utility helpers shared across training scripts."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import random
from typing import Any

import numpy as np
import torch


def create_directory(path: str | Path) -> Path:
    """Create a directory (recursively) if it does not exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def set_seed(seed: int = 42) -> None:
    """Set RNG seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Reproducible but may slow down
                torch.backends.cudnn.ofterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception:
        # torch may not be available in some minimal test environments
        pass


def timestamp(fmt: str = "%Y%m%d-%H%M%S") -> str:
    """Return a formatted timestamp string."""
    return datetime.now().strftime(fmt)


def save_json(data: Any, path: str | Path) -> None:
    """Persist JSON data ensuring parent directory exists."""
    path_obj = Path(path)
    create_directory(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load JSON content using UTF-8 encoding."""
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)
