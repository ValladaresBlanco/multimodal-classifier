"""
Preprocesador of images for clasificación
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
    Preprocesador of images siguiendo Nogle Responsibility Principle
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augmentation: bool = False):
        """
        Args:
            image_size: Size al que redimensionar (ancho, alto)
            normalize: Whether to apply ImageNet normalization
            augmentation: Si aplicar data augmentation
        """
        self.image_size = image_size
        self.normalize = normalize
        self.augmentation = augmentation
        
        # ImageNet statistics for normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Configure transformaciones
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Configurar transformaciones of albumentations"""
        
        # Transformaciones básicas (siempre se aplican)
        basic_transforms = [
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
        ]
        
        # Transformaciones of augmentation (solo en training)
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
        
        # Conseet a tensor PyTorch
        to_tensor_transforms = [ToTensorV2()]
        
        # Combinar todas las transformaciones
        all_transforms = basic_transforms + augmentation_transforms + normalize_transforms + to_tensor_transforms
        
        self.transform = A.Compose(all_transforms)
    
    def preprocess(self, image: np.ndarray):
        """
        Preprocess an image
        
        Args:
            image: Image in numpy format (H, W, C) BGR
            
        Returns:
            Tensor PyTorch (C, H, W) normalizado
        """
        # Conseet BGR a RGB (OpenCV carga en BGR)
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
        Preprocesar un lote of images
        
        Args:
            images: Lista of images numpy
            
        Returns:
            Array numpy of images preprocesadas (N, H, W, C)
        """
        processed = [self.preprocess(img) for img in images]
        return np.array(processed)
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image from disk
        
        Args:
            image_path: Path to the image
        
        Returns:
            Preprocessed image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return self.preprocess(image)


def test_preprocessor():
    """Función of test for el preprocesador"""
    print("🧪 Probando ImagePreprocessor...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f" Test image created: {test_image.shape}")
    
    # Probar preprocesador sin augmentation
    preprocessor = ImagePreprocessor(image_size=(224, 224), augmentation=False)
    processed = preprocessor.preprocess(test_image)
    print(f" Processed image: {processed.shape}")
    print(f" Rango of values: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Probar preprocesador with augmentation
    preprocessor_aug = ImagePreprocessor(image_size=(224, 224), augmentation=True)
    processed_aug = preprocessor_aug.preprocess(test_image)
    print(f" Image with augmentation: {processed_aug.shape}")
    
    # Probar batch
    batch = preprocessor.preprocess_batch([test_image, test_image])
    print(f" Batch procesado: {batch.shape}")
    
    print(" ImagePreprocessor running successfully!")


if __name__ == "__main__":
    test_preprocessor()
