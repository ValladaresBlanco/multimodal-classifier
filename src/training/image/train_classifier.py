"""
Main training script for image classifiers
Supports ResNet and MobileNet
"""

import torch
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.preprocessing.image_preprocessor import ImagePreprocessor
from src.data.loaders.image_loader import ImageDataLoader
from src.models.image_classifier.resnet_classifier import ResNetClassifier
from src.models.image_classifier.mobilenet_classifier import MobileNetClassifier
from src.utils.helpers import create_directory, set_seed


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    if history['val_acc']:
        ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def train_model(
    model_type: str = 'resnet',
    data_dir: str = 'data/raw/images',
    num_classes: int = None,
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 0.001,
    image_size: tuple = (224, 224),
    pretrained: bool = True,
    freeze_backbone: bool = False,
    save_dir: str = 'models/saved'
):
    """
    Train image classification model
    
    Args:
        model_type: 'resnet' or 'mobilenet'
        data_dir: Directory with images (structure: class/img.jpg)
        num_classes: Number of classes (None = auto-detect)
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        image_size: Image size (width, height)
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone
        save_dir: Directory to save models
    """
    
    print("=" * 70)
    print("IMAGE CLASSIFIER TRAINING")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create directories
    create_directory(save_dir)
    create_directory('results/visualizations')
    
    # 1. PREPARE DATA
    print("\nSTEP 1: Loading dataset...")
    print(f"   Directory: {data_dir}")
    
    # Create preprocessors
    preprocessor_train = ImagePreprocessor(
        image_size=image_size,
        normalize=True,
        augmentation=True  # Augmentation for training
    )
    
    preprocessor_val = ImagePreprocessor(
        image_size=image_size,
        normalize=True,
        augmentation=False  # No augmentation for validation
    )
    
    # Create data loader
    data_loader = ImageDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.2,
        seed=42
    )
    
    # Get class info
    data_loader.load_dataset()  # To extract classes
    class_info = data_loader.get_class_info()
    
    if num_classes is None:
        num_classes = class_info['num_classes']
    
    print(f"\nDataset information:")
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {class_info['class_names']}")
    
    # Create dataloaders
    train_loader, val_loader = data_loader.create_dataloaders(
        preprocessor_train, 
        preprocessor_val
    )
    
    # 2. CREATE MODEL
    print(f"\nSTEP 2: Building model {model_type.upper()}...")
    
    config = {
        'num_classes': num_classes,
        'pretrained': pretrained,
        'freeze_backbone': freeze_backbone
    }
    
    if model_type.lower() == 'resnet':
        model = ResNetClassifier(config)
    elif model_type.lower() == 'mobilenet':
        model = MobileNetClassifier(config)
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    
    model.build_model()
    
    # 3. TRAIN
    print(f"\nSTEP 3: Training...")
    
    save_path = f"{save_dir}/{model_type}_best.pth"
    
    history = model.train(
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        save_path=save_path
    )
    
    # 4. EVALUATE WITH COMPLETE METRICS
    print(f"\nSTEP 4: Final evaluation with complete metrics...")
    
    results = model.evaluate(
        val_loader, 
        class_names=class_info['class_names'],
        save_dir=f"results/evaluation/{model_type}"
    )
    print(f"\n   Validation accuracy: {results['accuracy']:.2f}%")
    
    # 5. VISUALIZE
    print(f"\nSTEP 5: Generating visualizations...")
    
    plot_path = f"results/visualizations/{model_type}_training.png"
    plot_training_history(history, save_path=plot_path)
    
    # 6. SAVE INFO
    print(f"\nSTEP 6: Saving information...")
    
    import json
    
    info = {
        'model_type': model_type,
        'num_classes': num_classes,
        'class_names': class_info['class_names'],
        'final_val_accuracy': results['accuracy'],
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'image_size': image_size,
        'pretrained': pretrained
    }
    
    info_path = f"{save_dir}/{model_type}_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"   Info saved to: {info_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"   Model: {save_path}")
    print(f"   Info: {info_path}")
    print(f"   Plot: {plot_path}")
    
    return model, history, results


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Train image classifier')
    
    parser.add_argument('--model', type=str, default='resnet', 
                       choices=['resnet', 'mobilenet'],
                       help='Model type')
    parser.add_argument('--data', type=str, default='data/raw/images',
                       help='Data directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Do not use pretrained weights')
    parser.add_argument('--freeze', action='store_true',
                       help='Freeze backbone')
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze
    )


if __name__ == "__main__":
    # If run directly without arguments, use default values
    if len(sys.argv) == 1:
        print("Running with default values...")
        print("   Use --help to see available options\n")
        
        train_model(
            model_type='resnet',
            data_dir='data/raw/images',
            epochs=5,  # Few epochs for quick test
            batch_size=8,
            lr=0.001,
            pretrained=True,
            freeze_backbone=False
        )
    else:
        main()
