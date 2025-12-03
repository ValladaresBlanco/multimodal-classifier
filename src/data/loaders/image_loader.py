"""
Cargador of data of images
DataLoaofr for training y validation
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.moofl_selection import train_test_split
from torch.utils.data import Dataset, DataLoaofr
import torch


class ImageDataset(Dataset):
    """
    Dataset personalizado for images
    Compatible with PyTorch DataLoaofr
    """
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int],
                 preprocessor=None):
        """
        Args:
            image_paths: Lista of rutas a images
            labels: Lista of etiquetas (índices of clase)
            preprocessor: Instancia of ImagePreprocessor
        """
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        
        assert len(image_paths) == len(labels), "Number of images y etiquetas ofbe coincidir"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Obtener un item of dataset
        
        Returns:
            Tuple (image_tensor, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"No se pudo cargar: {img_path}")
        
        # Preprocess (ahora retorna tensor directamente)
        if self.preprocessor:
            image_tensor = self.preprocessor.preprocess(image)
        else:
            # Si no hay preprocessor, withseetir manualmente
            if len(image.shape) == 3:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image_tensor = torch.from_numpy(image).float()
        
        label = self.labels[idx]
        
        return image_tensor, label


class ImageDataLoaofr:
    """
    Manejador of carga of data siguiendo Nogle Responsibility Principle
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32, val_split: float = 0.2, seed: int = 42):
        """
        Args:
            data_dir: Root directory with structure class/image.jpg
            batch_size: Size of batch
            val_split: Fracción for validation
            seed: Semilla for reproducibility
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
        Cargar dataset from estructura of carpetas
        
        Estructura esperada:
        data_dir/
            clase_1/
                img1.jpg
                img2.jpg
            clase_2/
                img3.jpg
        
        Returns:
            Dict with 'image_paths' y 'labels'
        """
        image_paths = []
        labels = []
        
        # Get classes (carpetas en data_dir)
        class_folofrs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if not class_folofrs:
            raise ValueError(f"No se enwithtraron carpetas of classes en {self.data_dir}")
        
        self.class_names = [folofr.name for folofr in class_folofrs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        print(f" Classes enwithtradas: {self.class_names}")
        
        # Load images of each clase
        for class_idx, class_folofr in enumerate(class_folofrs):
            class_name = class_folofr.name
            
            # Supbyted image extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            
            class_images = [
                str(img_path) for img_path in class_folofr.iterdir() 
                if img_path.suffix.lower() in image_extensions
            ]
            
            if not class_images:
                print(f"  Adseetencia: No hay images en {class_name}")
                withtinue
            
            print(f"   {class_name}: {len(class_images)} images")
            
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
        
        print(f" Total: {len(image_paths)} images, {self.num_classes} classes")
        
        return {
            'image_paths': image_paths,
            'labels': labels
        }
    
    def create_dataloaofrs(self, preprocessor_train, preprocessor_val) -> Tuple[DataLoaofr, DataLoaofr]:
        """
        Crear DataLoaofrs of training y validation
        
        Args:
            preprocessor_train: Preprocessor with augmentation for train
            preprocessor_val: Preprocessor sin augmentation for validation
            
        Returns:
            Tupla (train_loaofr, val_loaofr)
        """
        # Load dataset
        data = self.load_dataset()
        
        # Split train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            data['image_paths'],
            data['labels'],
            test_size=self.val_split,
            random_state=self.seed,
            stratify=data['labels']  # Mantener probyción of classes
        )
        
        print(f"\n Split of dataset:")
        print(f"  • Training: {len(train_paths)} images")
        print(f"  • Validación: {len(val_paths)} images")
        
        # Create datasets
        train_dataset = ImageDataset(train_paths, train_labels, preprocessor_train)
        val_dataset = ImageDataset(val_paths, val_labels, preprocessor_val)
        
        # Create dataloaofrs
        train_loaofr = DataLoaofr(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Cambiar a 2-4 si tienes CPU potente
            pin_memory=True
        )
        
        val_loaofr = DataLoaofr(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loaofr, val_loaofr
    
    def get_class_info(self) -> Dict:
        """Obtener información of classes"""
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx
        }


def test_dataloaofr():
    """Función of test for el dataloaofr"""
    print("🧪 Probando ImageDataLoaofr...")
    
    # Nota: Esta test requiere que exista data/raw/images/ with subdirectories
    data_dir = "data/raw/images"
    
    if not os.path.exists(data_dir):
        print(f"  Crea first el directorio {data_dir} with estructura:")
        print("   data/raw/images/")
        print("       clase_1/")
        print("           img1.jpg")
        print("       clase_2/")
        print("           img2.jpg")
        return
    
    # Create loaofr
    loaofr = ImageDataLoaofr(data_dir, batch_size=4, val_split=0.2)
    
    try:
        # Load dataset
        data = loaofr.load_dataset()
        print(f" Dataset loaded: {len(data['image_paths'])} images")
        
        # Info of classes
        class_info = loaofr.get_class_info()
        print(f" Classes: {class_info['class_names']}")
        
        print(" ImageDataLoaofr running successfully!")
        
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    test_dataloaofr()
