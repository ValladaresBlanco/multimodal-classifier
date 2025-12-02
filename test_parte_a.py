"""
Script de verificación rápida - Parte A
Prueba todos los componentes antes de entrenar
"""

import sys
import torch
import cv2
import numpy as np
from pathlib import Path

print("=" * 70)
print("🔍 VERIFICACIÓN DE COMPONENTES - PARTE A")
print("=" * 70)

# 1. Verificar PyTorch
print("\n1️⃣ Verificando PyTorch...")
print(f"   ✓ Versión: {torch.__version__}")
print(f"   ✓ CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")

# 2. Verificar OpenCV
print("\n2️⃣ Verificando OpenCV...")
print(f"   ✓ Versión: {cv2.__version__}")

# Test de cámara
print("   ⏳ Probando acceso a webcam...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"   ✓ Webcam OK - Resolución: {frame.shape}")
    cap.release()
else:
    print("   ⚠️  No se pudo acceder a la webcam")

# 3. Verificar estructura de directorios
print("\n3️⃣ Verificando estructura...")
data_dir = Path("data/raw/images")
if data_dir.exists():
    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    if classes:
        print(f"   ✓ Dataset encontrado con {len(classes)} clases: {classes}")
        
        # Contar imágenes
        total_images = 0
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                total_images += len(images)
                print(f"      • {class_dir.name}: {len(images)} imágenes")
        print(f"   ✓ Total: {total_images} imágenes")
        
        if total_images < 30:
            print(f"   ⚠️  Recomendado: al menos 50-70 imágenes")
    else:
        print(f"   ⚠️  {data_dir} existe pero no tiene clases (carpetas)")
        print("      Crea carpetas con nombres de clases y agrega imágenes")
else:
    print(f"   ❌ {data_dir} no existe")
    print("      Crea la estructura: data/raw/images/clase_1/img.jpg")

# 4. Verificar dependencias
print("\n4️⃣ Verificando dependencias...")
try:
    import albumentations
    print(f"   ✓ albumentations: {albumentations.__version__}")
except ImportError:
    print("   ❌ albumentations no instalado")
    print("      Instala: pip install albumentations")

try:
    from sklearn.model_selection import train_test_split
    print("   ✓ scikit-learn")
except ImportError:
    print("   ❌ scikit-learn no instalado")

try:
    import matplotlib
    print(f"   ✓ matplotlib: {matplotlib.__version__}")
except ImportError:
    print("   ❌ matplotlib no instalado")

# 5. Probar imports del proyecto
print("\n5️⃣ Verificando módulos del proyecto...")
errors = []

try:
    from src.data.preprocessing.image_preprocessor import ImagePreprocessor
    print("   ✓ ImagePreprocessor")
except Exception as e:
    print(f"   ❌ ImagePreprocessor: {e}")
    errors.append("ImagePreprocessor")

try:
    from src.data.loaders.image_loader import ImageDataLoader
    print("   ✓ ImageDataLoader")
except Exception as e:
    print(f"   ❌ ImageDataLoader: {e}")
    errors.append("ImageDataLoader")

try:
    from src.models.image_classifier.resnet_classifier import ResNetClassifier
    print("   ✓ ResNetClassifier")
except Exception as e:
    print(f"   ❌ ResNetClassifier: {e}")
    errors.append("ResNetClassifier")

try:
    from src.models.image_classifier.mobilenet_classifier import MobileNetClassifier
    print("   ✓ MobileNetClassifier")
except Exception as e:
    print(f"   ❌ MobileNetClassifier: {e}")
    errors.append("MobileNetClassifier")

# 6. Test rápido de funcionalidad
print("\n6️⃣ Probando funcionalidad básica...")
try:
    # Crear imagen de prueba
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    from src.data.preprocessing.image_preprocessor import ImagePreprocessor
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess(test_img)
    print(f"   ✓ Preprocesamiento OK - Shape: {processed.shape}")
    
    # Crear modelo de prueba
    from src.models.image_classifier.resnet_classifier import ResNetClassifier
    config = {'num_classes': 3, 'pretrained': False}
    model = ResNetClassifier(config)
    model.build_model()
    print("   ✓ Construcción de modelo OK")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model.predict(dummy_input)
    print(f"   ✓ Forward pass OK - Output: {output.shape}")
    
except Exception as e:
    print(f"   ❌ Error en test funcional: {e}")
    errors.append("test_funcional")

# Resumen final
print("\n" + "=" * 70)
if not errors:
    print("✅ VERIFICACIÓN COMPLETA - TODO OK!")
    print("\n🚀 Siguiente paso:")
    print("   1. Asegúrate de tener 50-70 imágenes en data/raw/images/")
    print("   2. Ejecuta: python src/training/image/train_classifier.py")
else:
    print("⚠️  VERIFICACIÓN INCOMPLETA")
    print(f"\nProblemas encontrados: {len(errors)}")
    for error in errors:
        print(f"   • {error}")
    print("\nRevisa los errores arriba y corrige antes de continuar")

print("=" * 70)
