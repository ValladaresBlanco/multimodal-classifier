"""Entrenamiento de modelo CTC para reconocimiento de voz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data.loaders.audio_loader import AudioDataModule
from src.models.audio_recognizer.ctc_speech_recognizer import CTCSpeechRecognizer
from src.utils import create_directory, save_json, set_seed, timestamp


def train_ctc_speech(
    metadata_csv: str = "data/raw/audio/metadata.csv",
    audio_root: str | None = "data/raw/audio",
    batch_size: int = 8,
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.2,
    grad_clip: float = 5.0,
    save_dir: str = "models/checkpoints",
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
):
    """Entrena un modelo speech-to-text basado en CTC."""

    print("=" * 72)
    print("🚀 ENTRENAMIENTO CTC (SPEECH-TO-TEXT)")
    print("=" * 72)

    set_seed(seed)
    audio_root_path = Path(audio_root) if audio_root else Path(metadata_csv).parent

    data_module = AudioDataModule(
        metadata_csv=metadata_csv,
        audio_root=audio_root_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )
    train_loader, val_loader, test_loader = data_module.dataloaders()

    config = {
        "vocab": data_module.tokenizer.vocab,
        "blank_symbol": data_module.tokenizer.blank_symbol,
        "feature_dim": data_module.preprocessor.feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "grad_clip": grad_clip,
    }

    model = CTCSpeechRecognizer(config)
    history = model.fit(train_loader, val_loader, epochs=epochs, lr=lr)
    val_loss, val_wer = model.evaluate(val_loader, compute_loss=True)

    test_metrics = None
    if test_loader is not None:
        test_metrics = model.evaluate(test_loader, compute_loss=True)

    create_directory(save_dir)
    run_id = timestamp()
    checkpoint_path = Path(save_dir) / f"ctc_speech_{run_id}.pth"
    model.save_model(str(checkpoint_path))

    info = {
        "metadata_csv": metadata_csv,
        "audio_root": str(audio_root_path),
        "sample_rate": data_module.preprocessor.config.sample_rate,
        "feature_dim": data_module.preprocessor.feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "grad_clip": grad_clip,
        "history": history,
        "val_loss": val_loss,
        "val_wer": val_wer,
        "test": {
            "loss": test_metrics[0],
            "wer": test_metrics[1],
        }
        if test_metrics
        else None,
        "checkpoint": str(checkpoint_path),
        "tokenizer": data_module.tokenizer.to_dict(),
        "preprocess": data_module.preprocessor.to_dict(),
    }

    save_path = Path(save_dir) / "ctc_speech_info.json"
    save_json(info, save_path)

    print("\n✅ Entrenamiento completado!")
    print(f"   • Checkpoint: {checkpoint_path}")
    print(f"   • Info: {save_path}")
    print(f"   • WER validación: {val_wer:.3f}")
    if test_metrics:
        print(f"   • WER test: {test_metrics[1]:.3f}")

    return model, history, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar modelo CTC para speech-to-text")
    parser.add_argument("--metadata", type=str, default="data/raw/audio/metadata.csv", help="Ruta al metadata CSV")
    parser.add_argument("--audio-root", type=str, default="data/raw/audio", help="Raíz de los audios")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--save-dir", type=str, default="models/checkpoints")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers para DataLoader (usar >0 acelera CPU)")
    parser.add_argument("--pin-memory", action="store_true", help="Activa pin_memory en DataLoader")
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Mantiene workers vivos entre batches (requiere num_workers>0)",
    )
    parser.add_argument("--max-train-samples", type=int, help="Limita muestras de train para sesiones rápidas")
    parser.add_argument("--max-val-samples", type=int, help="Limita muestras de validación")
    parser.add_argument("--max-test-samples", type=int, help="Limita muestras de test")
    return parser.parse_args()


def main():
    args = parse_args()
    train_ctc_speech(
        metadata_csv=args.metadata,
        audio_root=args.audio_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("💡 Ejecutando con parámetros por defecto (modo demo)...")
        try:
            train_ctc_speech(epochs=1)
        except Exception as exc:
            print(f"⚠️  No se pudo entrenar: {exc}")
            print("   Verifica que metadata.csv exista y que haya al menos 100 audios etiquetados.")
    else:
        main()
