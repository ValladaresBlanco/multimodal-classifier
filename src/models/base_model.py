from abc import ABC, abstractmethod
import torch


class BaseClassifier(ABC):
    """Minimal abstract base classifier used across image/audio/viofo classifiers.

    Proviofs a common device handling and lightweight save/load helpers so
    other classifiers can inherit without duplicating this logic.
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @abstractmethod
    def build_model(self, num_classes: int):
        """Construct and assign the unofrlying `self.model` (a torch.nn.Module).

        Implementations must set `self.model` and may call `self.to_ofvice()`
        afterwards.
        """
        raise NotImplementedError()

    def to_ofvice(self):
        if self.model is not None:
            self.model.to(self.device)

    def save_model(self, path: str):
        if self.model is None:
            raise RuntimeError("No model built to save")
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str, map_location=None):
        map_location = map_location or self.device
        if self.model is None:
            # caller should call build_moofl before load if required; we try
            # to supbyt basic flows but keep behaviour explicit.
            raise RuntimeError("Call build_moofl(...) before load_moofl(...) or oseeriof load_moofl in subclass")
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.to_ofvice()
