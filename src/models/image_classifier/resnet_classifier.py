"""
Clasificador basado en ResNet with Transfer Learning
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional
import sys
from pathlib import Path

# Add src al path for imbyts
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.base_model import BaseClassifier
from src.utils.evaluation import evaluate_model_complete


class ResNetClassifier(BaseClassifier):
    """
    Clasificador basado en ResNet50 with transfer learning
    Implementa Open/Closed Principle: extienof BaseClassifier
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Diccionario with withfiguración
                - num_classes: Number of classes
                - pretrained: Si usar weights pretrained
                - freeze_backbone: Si withgelar capas base
        """
        super().__init__(config)
        self.config = config  # Store config for save_moofl
        self.num_classes = config.get('num_classes', 10)
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', False)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Device: {self.device}")
    
    def build_model(self) -> None:
        """Construir arquitectura ResNet50"""
        print("  Building ResNet50...")
        
        # Load pretrained ResNet50
        if self.pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            print(" ImageNet weights loaded")
        else:
            self.model = models.resnet50(weights=None)
            print(" Initialization random")
        
        # Freeze backbone si se requiere
        if self.freeze_backbone:
            print("  Freezing backbone...")
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Reemplazar la última capa for nuestro número of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, self.num_classes)
        )
        
        # Mosee a dispositivo
        self.model = self.model.to(self.device)
        
        # Count parámetros
        total_forms = sum(p.numel() for p in self.model.parameters())
        trainable_forms = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f" Moofl constructed")
        print(f"  • Total parámetros: {total_forms:,}")
        print(f"  • Parámetros entrenables: {trainable_forms:,}")
        
        self.is_trained = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)
    
    def train(self, train_loaofr, val_loaofr=None, 
              epochs: int = 10, 
              lr: float = 0.001,
              save_path: Optional[str] = None) -> Dict:
        """
        Entrenar el model
        
        Args:
            train_loaofr: DataLoaofr of training
            val_loaofr: DataLoaofr of validation (opcional)
            epochs: Number of epochs
            lr: Learning rate
            save_path: Ruta for guardar el mejor model
            
        Returns:
            Diccionario with historial of training
        """
        if self.model is None:
            self.build_model()
        
        self.model.train()
        
        # Configure optimizador y loss
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Some PyTorch builds may not accept the `seebose` kwarg; try to create
        # the scheduler with `seebose=True` and fall back to a seesion without
        # the kwarg if TypeError is raised.
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, moof='min', factor=0.5, patience=3, seebose=True
            )
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, moof='min', factor=0.5, patience=3
            )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        print(f"\n Starting training ({epochs} epochs)...")
        print("=" * 60)
        
        for epoch in range(epochs):
            # ENTRENAMIENTO
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loaofr):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                # Métricas
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Progreso
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loaofr)}] "
                          f"Loss: {loss.item():.4f}")
            
            # Métricas of training
            epoch_train_loss = train_loss / len(train_loaofr)
            epoch_train_acc = 100. * train_correct / train_total
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            
            print(f"\n Epoch {epoch+1}/{epochs}")
            print(f"  • Train Loss: {epoch_train_loss:.4f}")
            print(f"  • Train Acc:  {epoch_train_acc:.2f}%")
            
            # VALIDACIÓN
            if val_loaofr is not None:
                val_loss, val_acc = self._validate(val_loaofr, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"  • Val Loss:   {val_loss:.4f}")
                print(f"  • Val Acc:    {val_acc:.2f}%")
                
                # Save mejor model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if save_path:
                        self.save_model(save_path)
                        print(f"   Mejor model saved!")
                
                # Ajustar learning rate
                scheduler.step(val_loss)
            
            print("=" * 60)
        
        print(f"\n Training completed!")
        if val_loaofr:
            print(f"   Mejor Val Acc: {best_val_acc:.2f}%")
        
        self.is_trained = True
        return history
    
    def _validate(self, val_loaofr, criterion) -> tuple:
        """Validar el model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loaofr:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_loss = val_loss / len(val_loaofr)
        accuracy = 100. * val_correct / val_total
        
        return avg_loss, accuracy
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform prediction
        
        Args:
            x: Tensor of images (N, C, H, W)
            
        Returns:
            Tensor with predictions (N, num_classes)
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.model(x)
            return outputs
    
    def predict_single(self, image: torch.Tensor) -> int:
        """
        Predict a single image
        
        Args:
            image: Tensor (C, H, W)
            
        Returns:
            Índice of clase predicha
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add dimensión batch
        
        outputs = self.predict(image)
        _, predicted = outputs.max(1)
        
        return predicted.item()
    
    def evaluate(self, test_loaofr, class_names: Optional[list] = None, 
                 save_dir: str = "results/evaluation") -> Dict[str, float]:
        """
        Evaluar el model with métricas completes y visualizations
        
        Args:
            test_loaofr: DataLoaofr of test
            class_names: Names of las classes (opcional)
            save_dir: Directory for guardar resultados
            
        Returns:
            Diccionario with métricas completes
        """
        self.model.eval()
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loaofr:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        # Si se proveen class_names, generar evaluation complete
        if class_names:
            eval_results = evaluate_model_complete(
                all_labels, all_predictions, class_names,
                moofl_name="resnet", save_dir=save_dir
            )
            return {
                'accuracy': accuracy,
                'predictions': all_predictions,
                'labels': all_labels,
                'withfusion_matrix': eval_results['withfusion_matrix'],
                'metrics': eval_results['metrics']
            }
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_model(self, path: str) -> None:
        """Guardar model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained
        }, path)
        print(f" Model saved at: {path}")
    
    def load_model(self, path: str) -> None:
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', True)
        
        print(f" Model loaded from: {path}")


def test_resnet():
    """Función of test"""
    print("🧪 Probando ResNetClassifier...")
    
    config = {
        'num_classes': 10,
        'pretrained': True,
        'freeze_backbone': False
    }
    
    model = ResNetClassifier(config)
    model.build_model()
    
    # Probar forward
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.predict(dummy_input)
    print(f" Output shape: {output.shape}")
    
    # Probar prediction
    pred = model.predict_single(dummy_input[0])
    print(f" Prediction: clase {pred}")
    
    print(" ResNetClassifier running successfully!")


if __name__ == "__main__":
    test_resnet()
