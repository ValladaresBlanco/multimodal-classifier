"""Real-time speech-to-text demo using the CTC model."""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocessing.audio_preprocessor import AudioPreprocessConfig, AudioPreprocessor
from src.data.preprocessing.text_tokenizer import TextTokenizer
from src.models.audio_recognizer.ctc_speech_recognizer import CTCSpeechRecognizer
from src.utils import load_json


class RealtimeCTCTranscriber:
    def __init__(
        self,
        model_path: str,
        info_path: str,
        window_seconds: float = 3.0,
        chunk_seconds: float = 0.5,
    ) -> None:
        self.info = load_json(info_path)
        preprocess_cfg = AudioPreprocessConfig(**self.info.get("preprocess", {}))
        self.preprocessor = AudioPreprocessor(preprocess_cfg)
        tokenizer = TextTokenizer.from_dict(self.info["tokenizer"])

        config = {
            "vocab": tokenizer.vocab,
            "blank_symbol": tokenizer.blank_symbol,
            "feature_dim": self.info.get("feature_dim", self.preprocessor.feature_dim),
            "hidden_dim": self.info.get("hidden_dim", 256),
            "num_layers": self.info.get("num_layers", 3),
            "dropout": self.info.get("dropout", 0.2),
            "lr": self.info.get("lr", 1e-3),
            "grad_clip": self.info.get("grad_clip", 5.0),
        }

        self.model = CTCSpeechRecognizer(config)
        self.model.load_model(model_path)

        self.tokenizer = tokenizer
        self.sample_rate = self.info.get("sample_rate", preprocess_cfg.sample_rate)
        self.window_samples = int(window_seconds * self.sample_rate)
        self.chunk_samples = int(chunk_seconds * self.sample_rate)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.queue: queue.Queue[np.ndarray] = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):  # type: ignore[override]
        if status:
            print(status, file=sys.stderr)
        self.queue.put(indata[:, 0].copy())

    def _drain_queue(self) -> bool:
        updated = False
        while not self.queue.empty():
            chunk = self.queue.get()
            self.buffer = np.concatenate([self.buffer, chunk])
            if self.buffer.size > self.window_samples * 2:
                self.buffer = self.buffer[-self.window_samples * 2 :]
            updated = True
        return updated

    def _transcribe_buffer(self) -> None:
        if self.buffer.size < self.window_samples:
            return
        window = self.buffer[-self.window_samples :]
        waveform = torch.from_numpy(window).unsqueeze(0)
        features, length = self.preprocessor.extract_features(waveform, apply_augment=False)
        features = features.unsqueeze(0)
        lengths = torch.tensor([length], dtype=torch.long)
        text = self.model.transcribe(features, lengths)[0]
        print(f"🗣️  {text}")

    def run(self) -> None:
        print("🎤 Transcripción en tiempo real - presiona Ctrl+C para salir")
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        ):
            try:
                while True:
                    if self._drain_queue():
                        self._transcribe_buffer()
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n👋 Saliendo del modo en vivo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcripción CTC en vivo")
    parser.add_argument("--model", required=True, help="Ruta al checkpoint .pth")
    parser.add_argument("--info", required=True, help="Ruta al archivo ctc_speech_info.json")
    parser.add_argument("--window", type=float, default=3.0, help="Segundos usados por ventana de inferencia")
    parser.add_argument("--chunk", type=float, default=0.5, help="Duración del chunk capturado")
    return parser.parse_args()


def main():
    args = parse_args()
    transcriber = RealtimeCTCTranscriber(
        model_path=args.model,
        info_path=args.info,
        window_seconds=args.window,
        chunk_seconds=args.chunk,
    )
    transcriber.run()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("💡 Uso:")
        print("   python -m src.app.realtime.audio_transcriber --model models/checkpoints/ctc_speech_xxx.pth --info models/checkpoints/ctc_speech_info.json")
    else:
        main()
