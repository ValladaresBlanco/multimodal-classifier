"""
Preprocesador de imágenes para clasificación
Normalización, redimensionamiento y data augmentation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImagePreprocessor:
    """
    Preprocesador de imágenes siguiendo Single Responsibility Principle
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augmentation: bool = False):
        """
        Args:
            image_size: Tamaño al que redimensionar (ancho, alto)
            normalize: Si aplicar normalización ImageNet
            augmentation: Si aplicar data augmentation
        """
        self.image_size = image_size
        self.normalize = normalize
        self.augmentation = augmentation
        
        # Estadísticas de ImageNet para normalización
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Configurar transformaciones
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Configurar transformaciones de albumentations"""
        
        # Transformaciones básicas (siempre se aplican)
        basic_transforms = [
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
        ]
        
        # Transformaciones de augmentation (solo en entrenamiento)
        augmentation_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.GaussNoise(p=0.2),
        ] if self.augmentation else []
        
        # Normalización (si está activada)
        normalize_transforms = [
            A.Normalize(mean=self.mean, std=self.std)
        ] if self.normalize else []
        
        # Convertir a tensor PyTorch
        to_tensor_transforms = [ToTensorV2()]
        
        # Combinar todas las transformaciones
        all_transforms = basic_transforms + augmentation_transforms + normalize_transforms + to_tensor_transforms
        
        self.transform = A.Compose(all_transforms)
    
    def preprocess(self, image: np.ndarray):
        """
        Preprocesar una imagen
        
        Args:
            image: Imagen en formato numpy (H, W, C) BGR
            
        Returns:
            Tensor PyTorch (C, H, W) normalizado
        """
        # Convertir BGR a RGB (OpenCV carga en BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Aplicar transformaciones (incluye ToTensorV2 que retorna tensor)
        transformed = self.transform(image=image_rgb)
        processed_image = transformed['image']
        
        return processed_image
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocesar un lote de imágenes
        
        Args:
            images: Lista de imágenes numpy
            
        Returns:
            Array numpy de imágenes preprocesadas (N, H, W, C)
        """
        processed = [self.preprocess(img) for img in images]
        return np.array(processed)
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Cargar y preprocesar una imagen desde disco
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Imagen preprocesada
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        return self.preprocess(image)


def test_preprocessor():
    """Función de prueba para el preprocesador"""
    print("🧪 Probando ImagePreprocessor...")
    
    # Crear imagen de prueba
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"✓ Imagen de prueba creada: {test_image.shape}")
    
    # Probar preprocesador sin augmentation
    preprocessor = ImagePreprocessor(image_size=(224, 224), augmentation=False)
    processed = preprocessor.preprocess(test_image)
    print(f"✓ Imagen procesada: {processed.shape}")
    print(f"✓ Rango de valores: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Probar preprocesador con augmentation
    preprocessor_aug = ImagePreprocessor(image_size=(224, 224), augmentation=True)
    processed_aug = preprocessor_aug.preprocess(test_image)
    print(f"✓ Imagen con augmentation: {processed_aug.shape}")
    
    # Probar batch
    batch = preprocessor.preprocess_batch([test_image, test_image])
    print(f"✓ Batch procesado: {batch.shape}")
    
    print("✅ ImagePreprocessor funciona correctamente!")


if __name__ == "__main__":
    test_preprocessor()
