"""
Clasificador basado en ResNet con Transfer Learning
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional
import sys
from pathlib import Path

# Agregar src al path para imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.base_model import BaseClassifier
from src.utils.evaluation import evaluate_model_complete


class ResNetClassifier(BaseClassifier):
    """
    Clasificador basado en ResNet50 con transfer learning
    Implementa Open/Closed Principle: extiende BaseClassifier
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Diccionario con configuración
                - num_classes: Número de clases
                - pretrained: Si usar pesos preentrenados
                - freeze_backbone: Si congelar capas base
        """
        super().__init__(config)
        self.config = config  # Store config for save_model
        self.num_classes = config.get('num_classes', 10)
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', False)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Dispositivo: {self.device}")
    
    def build_model(self) -> None:
        """Construir arquitectura ResNet50"""
        print("🏗️  Construyendo ResNet50...")
        
        # Cargar ResNet50 preentrenado
        if self.pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            print("✓ Pesos ImageNet cargados")
        else:
            self.model = models.resnet50(weights=None)
            print("✓ Inicialización aleatoria")
        
        # Congelar backbone si se requiere
        if self.freeze_backbone:
            print("❄️  Congelando backbone...")
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Reemplazar la última capa para nuestro número de clases
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, self.num_classes)
        )
        
        # Mover a dispositivo
        self.model = self.model.to(self.device)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Modelo construido")
        print(f"  • Total parámetros: {total_params:,}")
        print(f"  • Parámetros entrenables: {trainable_params:,}")
        
        self.is_trained = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)
    
    def train(self, train_loader, val_loader=None, 
              epochs: int = 10, 
              lr: float = 0.001,
              save_path: Optional[str] = None) -> Dict:
        """
        Entrenar el modelo
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            epochs: Número de épocas
            lr: Learning rate
            save_path: Ruta para guardar el mejor modelo
            
        Returns:
            Diccionario con historial de entrenamiento
        """
        if self.model is None:
            self.build_model()
        
        self.model.train()
        
        # Configurar optimizador y loss
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Some PyTorch builds may not accept the `verbose` kwarg; try to create
        # the scheduler with `verbose=True` and fall back to a version without
        # the kwarg if TypeError is raised.
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
        
        print(f"\n🚀 Iniciando entrenamiento ({epochs} épocas)...")
        print("=" * 60)
        
        for epoch in range(epochs):
            # ENTRENAMIENTO
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
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
                    print(f"  Época [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")
            
            # Métricas de entrenamiento
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100. * train_correct / train_total
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            
            print(f"\n📊 Época {epoch+1}/{epochs}")
            print(f"  • Train Loss: {epoch_train_loss:.4f}")
            print(f"  • Train Acc:  {epoch_train_acc:.2f}%")
            
            # VALIDACIÓN
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"  • Val Loss:   {val_loss:.4f}")
                print(f"  • Val Acc:    {val_acc:.2f}%")
                
                # Guardar mejor modelo
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if save_path:
                        self.save_model(save_path)
                        print(f"  ✓ Mejor modelo guardado!")
                
                # Ajustar learning rate
                scheduler.step(val_loss)
            
            print("=" * 60)
        
        print(f"\n✅ Entrenamiento completado!")
        if val_loader:
            print(f"   Mejor Val Acc: {best_val_acc:.2f}%")
        
        self.is_trained = True
        return history
    
    def _validate(self, val_loader, criterion) -> tuple:
        """Validar el modelo"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * val_correct / val_total
        
        return avg_loss, accuracy
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realizar predicción
        
        Args:
            x: Tensor de imágenes (N, C, H, W)
            
        Returns:
            Tensor con predicciones (N, num_classes)
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.model(x)
            return outputs
    
    def predict_single(self, image: torch.Tensor) -> int:
        """
        Predecir una sola imagen
        
        Args:
            image: Tensor (C, H, W)
            
        Returns:
            Índice de clase predicha
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Agregar dimensión batch
        
        outputs = self.predict(image)
        _, predicted = outputs.max(1)
        
        return predicted.item()
    
    def evaluate(self, test_loader, class_names: Optional[list] = None, 
                 save_dir: str = "results/evaluation") -> Dict[str, float]:
        """
        Evaluar el modelo con métricas completas y visualizaciones
        
        Args:
            test_loader: DataLoader de test
            class_names: Nombres de las clases (opcional)
            save_dir: Directorio para guardar resultados
            
        Returns:
            Diccionario con métricas completas
        """
        self.model.eval()
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        # Si se proveen class_names, generar evaluación completa
        if class_names:
            eval_results = evaluate_model_complete(
                all_labels, all_predictions, class_names,
                model_name="resnet", save_dir=save_dir
            )
            return {
                'accuracy': accuracy,
                'predictions': all_predictions,
                'labels': all_labels,
                'confusion_matrix': eval_results['confusion_matrix'],
                'metrics': eval_results['metrics']
            }
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_model(self, path: str) -> None:
        """Guardar modelo"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained
        }, path)
        print(f"💾 Modelo guardado en: {path}")
    
    def load_model(self, path: str) -> None:
        """Cargar modelo"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', True)
        
        print(f"📥 Modelo cargado desde: {path}")


def test_resnet():
    """Función de prueba"""
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
    print(f"✓ Output shape: {output.shape}")
    
    # Probar predicción
    pred = model.predict_single(dummy_input[0])
    print(f"✓ Predicción: clase {pred}")
    
    print("✅ ResNetClassifier funciona correctamente!")


if __name__ == "__main__":
    test_resnet()
