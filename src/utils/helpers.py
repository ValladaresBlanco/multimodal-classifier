from pathlib import Path
import random
import numpy as np
import torch


def create_directory(path: str):
    """Create directory including parents if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42):
    """Set random seed for python, numpy and torch for reproducibility."""
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
