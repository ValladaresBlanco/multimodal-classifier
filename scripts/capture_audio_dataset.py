"""Herramienta CLI para grabar audios etiquetados para el dataset CTC."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable
import sys

import sounddevice as sd
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import create_directory


class AudioDatasetRecorder:
    def __init__(
        self,
        audio_root: str = "data/raw/audio",
        metadata_csv: str = "data/raw/audio/metadata.csv",
        sample_rate: int = 16_000,
    ) -> None:
        self.audio_root = Path(audio_root)
        self.metadata_csv = Path(metadata_csv)
        self.sample_rate = sample_rate
        create_directory(self.audio_root)
        create_directory(self.metadata_csv.parent)
        if not self.metadata_csv.exists():
            with self.metadata_csv.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=["path", "text", "split"])
                writer.writeheader()

    def record(self, transcript: str, duration: float = 3.0, split: str = "train") -> Path:
        transcript = transcript.strip()
        if not transcript:
            raise ValueError("El texto de la transcripción no puede estar vacío")
        file_name = f"sample_{int(time.time() * 1000)}.wav"
        target_dir = create_directory(self.audio_root / split)
        file_path = target_dir / file_name

        print(f"\n🎙️  Lee en voz alta: '{transcript}'")
        print("   Grabando en 3...", end="", flush=True)
        time.sleep(1)
        for count in (2, 1):
            print(f" {count}...", end="", flush=True)
            time.sleep(1)
        print(" ¡Ahora!")

        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()
        sf.write(file_path, recording, self.sample_rate)

        rel_path = file_path.relative_to(self.audio_root)
        with self.metadata_csv.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["path", "text", "split"])
            writer.writerow({"path": str(rel_path), "text": transcript, "split": split})

        print(f"   ✓ Guardado en {file_path}")
        return file_path

    def bulk_record(self, transcripts: Iterable[str], duration: float, split: str) -> None:
        for text in transcripts:
            self.record(text, duration=duration, split=split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graba audios etiquetados para el dataset")
    parser.add_argument("--metadata", default="data/raw/audio/metadata.csv")
    parser.add_argument("--audio-root", default="data/raw/audio")
    parser.add_argument("--duration", type=float, default=3.0, help="Duración de cada clip en segundos")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--phrases", type=str, nargs="*", help="Frases predeterminadas a grabar")
    return parser.parse_args()


def main():
    args = parse_args()
    recorder = AudioDatasetRecorder(audio_root=args.audio_root, metadata_csv=args.metadata)

    if args.phrases:
        recorder.bulk_record(args.phrases, duration=args.duration, split=args.split)
        return

    print("\n👉 Ingresa frases (ENTER vacío para terminar)")
    while True:
        try:
            text = input("Texto a grabar: ").strip()
        except EOFError:
            break
        if not text:
            break
        recorder.record(text, duration=args.duration, split=args.split)

    print("\n✅ Sesión completada. Revisa metadata.csv para ver el resumen.")


if __name__ == "__main__":
    main()
