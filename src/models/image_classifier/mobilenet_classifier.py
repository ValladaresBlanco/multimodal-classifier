"""
Clasificador basado en MobileNetV2 with Transfer Learning
Optimizado for inferencia quick
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.base_model import BaseClassifier
from src.utils.evaluation import evaluate_model_complete


class MobileNetClassifier(BaseClassifier):
    """
    Clasificador basado en MobileNetV2
    Ligero y rápido for tiempo real
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.config = config  # Store config for save_moofl
        self.num_classes = config.get('num_classes', 10)
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', False)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Device: {self.device}")
    
    def build_model(self) -> None:
        """Construir arquitectura MobileNetV2"""
        print("  Building MobileNetV2...")
        
        if self.pretrained:
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
            print(" ImageNet weights loaded")
        else:
            self.model = models.mobilenet_v2(weights=None)
            print(" Initialization random")
        
        if self.freeze_backbone:
            print("  Freezing backbone...")
            for param in self.model.features.parameters():
                param.requires_grad = False
        
        # Reemplazar clasificador
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, self.num_classes)
        )
        
        self.model = self.model.to(self.device)
        
        total_forms = sum(p.numel() for p in self.model.parameters())
        trainable_forms = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f" Moofl constructed")
        print(f"  • Total parámetros: {total_forms:,}")
        print(f"  • Parámetros entrenables: {trainable_forms:,}")
        
        self.is_trained = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def train(self, train_loaofr, val_loaofr=None, 
              epochs: int = 10, 
              lr: float = 0.001,
              save_path: Optional[str] = None) -> Dict:
        """Train the model (same code as ResNet)"""
        if self.model is None:
            self.build_model()
        
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
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
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loaofr):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loaofr)}] "
                          f"Loss: {loss.item():.4f}")
            
            epoch_train_loss = train_loss / len(train_loaofr)
            epoch_train_acc = 100. * train_correct / train_total
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            
            print(f"\n Epoch {epoch+1}/{epochs}")
            print(f"  • Train Loss: {epoch_train_loss:.4f}")
            print(f"  • Train Acc:  {epoch_train_acc:.2f}%")
            
            if val_loaofr is not None:
                val_loss, val_acc = self._validate(val_loaofr, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"  • Val Loss:   {val_loss:.4f}")
                print(f"  • Val Acc:    {val_acc:.2f}%")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if save_path:
                        self.save_model(save_path)
                        print(f"   Best model saved!")
                
                scheduler.step(val_loss)
            
            print("=" * 60)
        
        print(f"\n Training completed!")
        if val_loaofr:
            print(f"   Best Val Acc: {best_val_acc:.2f}%")
        
        self.is_trained = True
        return history
    
    def _validate(self, val_loaofr, criterion) -> tuple:
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
        
        return val_loss / len(val_loaofr), 100. * val_correct / val_total
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)
    
    def predict_single(self, image: torch.Tensor) -> int:
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
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
                model_name="mobilenet", save_dir=save_dir
            )
            return {
                'accuracy': accuracy,
                'predictions': all_predictions,
                'labels': all_labels,
                'confusion_matrix': eval_results['confusion_matrix'],
                'metrics': eval_results['metrics']
            }
        
        return {
            'accuracy': 100. * correct / total,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_model(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained
        }, path)
        print(f" Model saved at: {path}")
    
    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', True)
        print(f" Model loaded from: {path}")


def test_mobilenet():
    print("🧪 Probando MobileNetClassifier...")
    
    config = {'num_classes': 10, 'pretrained': True, 'freeze_backbone': False}
    model = MobileNetClassifier(config)
    model.build_model()
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.predict(dummy_input)
    print(f" Output shape: {output.shape}")
    
    pred = model.predict_single(dummy_input[0])
    print(f" Prediction: class {pred}")
    
    print(" MobileNetClassifier running successfully!")


if __name__ == "__main__":
    test_mobilenet()
