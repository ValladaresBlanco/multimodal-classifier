"""
Cargador de datos de imágenes
DataLoader para entrenamiento y validación
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


class ImageDataset(Dataset):
    """
    Dataset personalizado para imágenes
    Compatible con PyTorch DataLoader
    """
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int],
                 preprocessor=None):
        """
        Args:
            image_paths: Lista de rutas a imágenes
            labels: Lista de etiquetas (índices de clase)
            preprocessor: Instancia de ImagePreprocessor
        """
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        
        assert len(image_paths) == len(labels), "Número de imágenes y etiquetas debe coincidir"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Obtener un item del dataset
        
        Returns:
            Tupla (imagen_tensor, etiqueta)
        """
        # Cargar imagen
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"No se pudo cargar: {img_path}")
        
        # Preprocesar (ahora retorna tensor directamente)
        if self.preprocessor:
            image_tensor = self.preprocessor.preprocess(image)
        else:
            # Si no hay preprocessor, convertir manualmente
            if len(image.shape) == 3:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image_tensor = torch.from_numpy(image).float()
        
        label = self.labels[idx]
        
        return image_tensor, label


class ImageDataLoader:
    """
    Manejador de carga de datos siguiendo Single Responsibility Principle
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32, val_split: float = 0.2, seed: int = 42):
        """
        Args:
            data_dir: Directorio raíz con estructura clase/imagen.jpg
            batch_size: Tamaño del batch
            val_split: Fracción para validación
            seed: Semilla para reproducibilidad
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        
        self.class_names = []
        self.class_to_idx = {}
        self.num_classes = 0
        
    def load_dataset(self) -> Dict[str, List]:
        """
        Cargar dataset desde estructura de carpetas
        
        Estructura esperada:
        data_dir/
            clase_1/
                img1.jpg
                img2.jpg
            clase_2/
                img3.jpg
        
        Returns:
            Dict con 'image_paths' y 'labels'
        """
        image_paths = []
        labels = []
        
        # Obtener clases (carpetas en data_dir)
        class_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if not class_folders:
            raise ValueError(f"No se encontraron carpetas de clases en {self.data_dir}")
        
        self.class_names = [folder.name for folder in class_folders]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        print(f"📁 Clases encontradas: {self.class_names}")
        
        # Cargar imágenes de cada clase
        for class_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            
            # Extensiones de imagen soportadas
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            
            class_images = [
                str(img_path) for img_path in class_folder.iterdir() 
                if img_path.suffix.lower() in image_extensions
            ]
            
            if not class_images:
                print(f"⚠️  Advertencia: No hay imágenes en {class_name}")
                continue
            
            print(f"  ✓ {class_name}: {len(class_images)} imágenes")
            
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
        
        print(f"📊 Total: {len(image_paths)} imágenes, {self.num_classes} clases")
        
        return {
            'image_paths': image_paths,
            'labels': labels
        }
    
    def create_dataloaders(self, preprocessor_train, preprocessor_val) -> Tuple[DataLoader, DataLoader]:
        """
        Crear DataLoaders de entrenamiento y validación
        
        Args:
            preprocessor_train: Preprocessor con augmentation para train
            preprocessor_val: Preprocessor sin augmentation para validación
            
        Returns:
            Tupla (train_loader, val_loader)
        """
        # Cargar dataset
        data = self.load_dataset()
        
        # Split train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            data['image_paths'],
            data['labels'],
            test_size=self.val_split,
            random_state=self.seed,
            stratify=data['labels']  # Mantener proporción de clases
        )
        
        print(f"\n📊 Split del dataset:")
        print(f"  • Entrenamiento: {len(train_paths)} imágenes")
        print(f"  • Validación: {len(val_paths)} imágenes")
        
        # Crear datasets
        train_dataset = ImageDataset(train_paths, train_labels, preprocessor_train)
        val_dataset = ImageDataset(val_paths, val_labels, preprocessor_val)
        
        # Crear dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Cambiar a 2-4 si tienes CPU potente
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_class_info(self) -> Dict:
        """Obtener información de clases"""
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx
        }


def test_dataloader():
    """Función de prueba para el dataloader"""
    print("🧪 Probando ImageDataLoader...")
    
    # Nota: Esta prueba requiere que exista data/raw/images/ con subdirectorios
    data_dir = "data/raw/images"
    
    if not os.path.exists(data_dir):
        print(f"⚠️  Crea primero el directorio {data_dir} con estructura:")
        print("   data/raw/images/")
        print("       clase_1/")
        print("           img1.jpg")
        print("       clase_2/")
        print("           img2.jpg")
        return
    
    # Crear loader
    loader = ImageDataLoader(data_dir, batch_size=4, val_split=0.2)
    
    try:
        # Cargar dataset
        data = loader.load_dataset()
        print(f"✓ Dataset cargado: {len(data['image_paths'])} imágenes")
        
        # Info de clases
        class_info = loader.get_class_info()
        print(f"✓ Clases: {class_info['class_names']}")
        
        print("✅ ImageDataLoader funciona correctamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_dataloader()
