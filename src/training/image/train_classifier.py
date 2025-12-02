"""
Script principal de entrenamiento para clasificadores de imágenes
Soporta ResNet y MobileNet
"""

import torch
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.preprocessing.image_preprocessor import ImagePreprocessor
from src.data.loaders.image_loader import ImageDataLoader
from src.models.image_classifier.resnet_classifier import ResNetClassifier
from src.models.image_classifier.mobilenet_classifier import MobileNetClassifier
from src.utils.helpers import create_directory, set_seed


def plot_training_history(history, save_path=None):
    """Graficar historial de entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.set_title('Pérdida durante entrenamiento')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    if history['val_acc']:
        ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Precisión durante entrenamiento')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Gráfica guardada en: {save_path}")
    
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
    Entrenar modelo de clasificación de imágenes
    
    Args:
        model_type: 'resnet' o 'mobilenet'
        data_dir: Directorio con imágenes (estructura: clase/img.jpg)
        num_classes: Número de clases (None = autodetectar)
        batch_size: Tamaño del batch
        epochs: Número de épocas
        lr: Learning rate
        image_size: Tamaño de imagen (ancho, alto)
        pretrained: Usar pesos preentrenados
        freeze_backbone: Congelar backbone
        save_dir: Directorio para guardar modelos
    """
    
    print("=" * 70)
    print("🚀 ENTRENAMIENTO DE CLASIFICADOR DE IMÁGENES")
    print("=" * 70)
    
    # Configurar semilla para reproducibilidad
    set_seed(42)
    
    # Crear directorios
    create_directory(save_dir)
    create_directory('results/visualizations')
    
    # 1. PREPARAR DATOS
    print("\n📁 PASO 1: Cargando dataset...")
    print(f"   Directorio: {data_dir}")
    
    # Crear preprocessors
    preprocessor_train = ImagePreprocessor(
        image_size=image_size,
        normalize=True,
        augmentation=True  # Augmentation para entrenamiento
    )
    
    preprocessor_val = ImagePreprocessor(
        image_size=image_size,
        normalize=True,
        augmentation=False  # Sin augmentation para validación
    )
    
    # Crear data loader
    data_loader = ImageDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.2,
        seed=42
    )
    
    # Obtener info de clases
    data_loader.load_dataset()  # Para extraer clases
    class_info = data_loader.get_class_info()
    
    if num_classes is None:
        num_classes = class_info['num_classes']
    
    print(f"\n📊 Información del dataset:")
    print(f"   • Número de clases: {num_classes}")
    print(f"   • Clases: {class_info['class_names']}")
    
    # Crear dataloaders
    train_loader, val_loader = data_loader.create_dataloaders(
        preprocessor_train, 
        preprocessor_val
    )
    
    # 2. CREAR MODELO
    print(f"\n🏗️  PASO 2: Construyendo modelo {model_type.upper()}...")
    
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
        raise ValueError(f"Modelo no soportado: {model_type}")
    
    model.build_model()
    
    # 3. ENTRENAR
    print(f"\n🎯 PASO 3: Entrenamiento...")
    
    save_path = f"{save_dir}/{model_type}_best.pth"
    
    history = model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        save_path=save_path
    )
    
    # 4. EVALUAR CON MÉTRICAS COMPLETAS
    print(f"\n📈 PASO 4: Evaluación final con métricas completas...")
    
    results = model.evaluate(
        val_loader, 
        class_names=class_info['class_names'],
        save_dir=f"results/evaluation/{model_type}"
    )
    print(f"\n   ✅ Accuracy en validación: {results['accuracy']:.2f}%")
    
    # 5. VISUALIZAR
    print(f"\n📊 PASO 5: Generando visualizaciones...")
    
    plot_path = f"results/visualizations/{model_type}_training.png"
    plot_training_history(history, save_path=plot_path)
    
    # 6. GUARDAR INFO
    print(f"\n💾 PASO 6: Guardando información...")
    
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
    
    print(f"   ✓ Info guardada en: {info_path}")
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO!")
    print("=" * 70)
    print(f"\n📦 Archivos generados:")
    print(f"   • Modelo: {save_path}")
    print(f"   • Info: {info_path}")
    print(f"   • Gráfica: {plot_path}")
    
    return model, history, results


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Entrenar clasificador de imágenes')
    
    parser.add_argument('--model', type=str, default='resnet', 
                       choices=['resnet', 'mobilenet'],
                       help='Tipo de modelo')
    parser.add_argument('--data', type=str, default='data/raw/images',
                       help='Directorio de datos')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='No usar pesos preentrenados')
    parser.add_argument('--freeze', action='store_true',
                       help='Congelar backbone')
    
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
    # Si se ejecuta directamente sin argumentos, usar valores por defecto
    if len(sys.argv) == 1:
        print("💡 Ejecutando con valores por defecto...")
        print("   Usa --help para ver opciones disponibles\n")
        
        train_model(
            model_type='resnet',
            data_dir='data/raw/images',
            epochs=5,  # Pocas épocas para prueba rápida
            batch_size=8,
            lr=0.001,
            pretrained=True,
            freeze_backbone=False
        )
    else:
        main()
