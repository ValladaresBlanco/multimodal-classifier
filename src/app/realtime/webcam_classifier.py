"""
Clasificación en tiempo real con webcam
Usa modelo entrenado para clasificar video en vivo
"""

import cv2
import torch
import numpy as np
import sys
from pathlib import Path
import time
import json

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.preprocessing.image_preprocessor import ImagePreprocessor
from src.models.image_classifier.resnet_classifier import ResNetClassifier
from src.models.image_classifier.mobilenet_classifier import MobileNetClassifier


class WebcamClassifier:
    """
    Clasificador en tiempo real usando webcam
    """
    
    def __init__(self, 
                 model_path: str,
                 model_info_path: str,
                 confidence_threshold: float = 0.5):
        """
        Args:
            model_path: Ruta al modelo guardado (.pth)
            model_info_path: Ruta al archivo info.json
            confidence_threshold: Umbral de confianza para mostrar predicción
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Cargar información del modelo
        print("📥 Cargando información del modelo...")
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        self.model_type = self.model_info['model_type']
        self.num_classes = self.model_info['num_classes']
        self.class_names = self.model_info['class_names']
        
        print(f"   • Tipo: {self.model_type}")
        print(f"   • Clases: {self.class_names}")
        
        # Crear preprocesador
        self.preprocessor = ImagePreprocessor(
            image_size=(224, 224),
            normalize=True,
            augmentation=False
        )
        
        # Cargar modelo
        print("🔧 Cargando modelo...")
        config = {
            'num_classes': self.num_classes,
            'pretrained': False
        }
        
        if self.model_type == 'resnet':
            self.model = ResNetClassifier(config)
        elif self.model_type == 'mobilenet':
            self.model = MobileNetClassifier(config)
        else:
            raise ValueError(f"Modelo no soportado: {self.model_type}")
        
        self.model.load_model(model_path)
        self.model.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"✅ Modelo cargado y listo!")
        print(f"   Dispositivo: {self.device}")
    
    def predict_frame(self, frame: np.ndarray) -> tuple:
        """
        Predecir clase de un frame
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            (clase_idx, clase_nombre, confianza)
        """
        # Preprocesar
        processed = self.preprocessor.preprocess(frame)
        
        # Convertir a tensor
        image_tensor = torch.from_numpy(processed).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predecir
        with torch.no_grad():
            outputs = self.model.predict(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        class_idx = predicted.item()
        class_name = self.class_names[class_idx]
        conf = confidence.item()
        
        return class_idx, class_name, conf
    
    def run(self, camera_id: int = 0, window_name: str = "Clasificador en Tiempo Real"):
        """
        Ejecutar clasificación en tiempo real
        
        Args:
            camera_id: ID de la cámara (0 para webcam default)
            window_name: Nombre de la ventana
        """
        print("\n" + "=" * 60)
        print("🎥 CLASIFICACIÓN EN TIEMPO REAL")
        print("=" * 60)
        print("📹 Abriendo cámara...")
        
        # Abrir cámara
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Cámara iniciada!")
        print("\n⌨️  Controles:")
        print("   • ESC o 'q' - Salir")
        print("   • ESPACIO - Tomar captura")
        print("=" * 60 + "\n")
        
        # Variables para FPS
        fps_time = time.time()
        fps = 0
        frame_count = 0
        
        # Variables para captura
        capture_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error al leer frame")
                    break
                
                # Predecir
                class_idx, class_name, confidence = self.predict_frame(frame)
                
                # Calcular FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_time)
                    fps_time = time.time()
                
                # Dibujar resultados en el frame
                h, w = frame.shape[:2]
                
                # Fondo semitransparente para texto
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                
                # Texto de predicción
                if confidence >= self.confidence_threshold:
                    text = f"Clase: {class_name}"
                    color = (0, 255, 0)  # Verde si confianza alta
                else:
                    text = f"Clase: {class_name} (?)"
                    color = (0, 165, 255)  # Naranja si confianza baja
                
                cv2.putText(frame, text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Confianza
                conf_text = f"Confianza: {confidence*100:.1f}%"
                cv2.putText(frame, conf_text, (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # FPS
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Instrucciones
                cv2.putText(frame, "ESC: Salir | ESPACIO: Capturar", 
                           (20, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Barra de confianza
                bar_width = int((w - 40) * confidence)
                cv2.rectangle(frame, (20, h - 60), (20 + bar_width, h - 40), 
                             (0, 255, 0), -1)
                cv2.rectangle(frame, (20, h - 60), (w - 20, h - 40), 
                             (255, 255, 255), 2)
                
                # Mostrar
                cv2.imshow(window_name, frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC o 'q'
                    print("\n👋 Saliendo...")
                    break
                elif key == ord(' '):  # ESPACIO
                    capture_count += 1
                    filename = f"capture_{capture_count}_{class_name}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Captura guardada: {filename}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrumpido por usuario")
        
        finally:
            # Liberar recursos
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Recursos liberados")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clasificación en tiempo real con webcam')
    parser.add_argument('--model', type=str, required=True,
                       help='Ruta al modelo (.pth)')
    parser.add_argument('--info', type=str, required=True,
                       help='Ruta al archivo info.json')
    parser.add_argument('--camera', type=int, default=0,
                       help='ID de la cámara')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Umbral de confianza')
    
    args = parser.parse_args()
    
    classifier = WebcamClassifier(
        model_path=args.model,
        model_info_path=args.info,
        confidence_threshold=args.threshold
    )
    
    classifier.run(camera_id=args.camera)


if __name__ == "__main__":
    # Ejemplo de uso
    if len(sys.argv) == 1:
        print("💡 Uso:")
        print("   python webcam_classifier.py --model models/saved/resnet_best.pth --info models/saved/resnet_info.json")
        print("\n   Primero debes entrenar un modelo con train_classifier.py")
    else:
        main()
