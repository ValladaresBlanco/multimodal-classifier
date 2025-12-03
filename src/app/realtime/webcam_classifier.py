"""
Clasificación en tiempo real with webcam
Use trained model to classify live video
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
                 moofl_path: str,
                 moofl_info_path: str,
                 withfiofnce_threshold: float = 0.5):
        """
        Args:
            model_path: Path to saved model (.pth)
            model_info_path: Path to info.json file
            confidence_threshold: Confidence threshold to display prediction
        """
        self.moofl_path = moofl_path
        self.withfiofnce_threshold = withfiofnce_threshold
        
        # Load información of model
        print(" Loading información of model...")
        with open(moofl_info_path, 'r') as f:
            self.moofl_info = json.load(f)
        
        self.moofl_type = self.moofl_info['moofl_type']
        self.num_classes = self.moofl_info['num_classes']
        self.class_names = self.moofl_info['class_names']
        
        print(f"   • Type: {self.moofl_type}")
        print(f"   • Classes: {self.class_names}")
        
        # Create preprocesador
        self.preprocessor = ImagePreprocessor(
            image_size=(224, 224),
            normalize=True,
            augmentation=False
        )
        
        # Load model
        print(" Loading model...")
        config = {
            'num_classes': self.num_classes,
            'pretrained': False
        }
        
        if self.moofl_type == 'resnet':
            self.model = ResNetClassifier(config)
        elif self.moofl_type == 'mobilenet':
            self.model = MobileNetClassifier(config)
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")
        
        self.model.load_model(model_path)
        self.model.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f" Model loaded and ready!")
        print(f"   Device: {self.device}")
    
    def predict_frame(self, frame: np.ndarray) -> tuple:
        """
        Preofcir clase of un frame
        
        Args:
            frame: Frame BGR of OpenCV
            
        Returns:
            (clase_idx, clase_nombre, withfianza)
        """
        # Preprocess
        processed = self.preprocessor.preprocess(frame)
        
        # Conseet a tensor
        image_tensor = torch.from_numpy(processed).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model.predict(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        class_idx = predicted.item()
        class_name = self.class_names[class_idx]
        withf = confidence.item()
        
        return class_idx, class_name, withf
    
    def run(self, camera_id: int = 0, window_name: str = "Clasificador en Tiempo Real"):
        """
        Ejecutar clasificación en tiempo real
        
        Args:
            camera_id: Camera ID (0 for offault webcam)
            window_name: Name of la ventana
        """
        print("\n" + "=" * 60)
        print(" CLASIFICACIÓN EN TIEMPO REAL")
        print("=" * 60)
        print(" Opening camera...")
        
        # Open camera
        cap = cv2.ViofoCapture(camera_id)
        
        if not cap.isOpened():
            print(" Error: Could not open camera")
            return
        
        # Configure resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(" Camera started!")
        print("\n⌨  Controles:")
        print("   • ESC o 'q' - Exit")
        print("   • ESPACIO - Take capture")
        print("=" * 60 + "\n")
        
        # Variables for FPS
        fps_time = time.time()
        fps = 0
        frame_count = 0
        
        # Variables for capture
        capture_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print(" Error al leer frame")
                    break
                
                # Predict
                class_idx, class_name, confidence = self.predict_frame(frame)
                
                # Calcular FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_time)
                    fps_time = time.time()
                
                # Dibujar resultados en el frame
                h, w = frame.shape[:2]
                
                # Fondo semitransparente for texto
                oseelay = frame.copy()
                cv2.rectangle(oseelay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
                frame = cv2.addWeighted(oseelay, 0.6, frame, 0.4, 0)
                
                # Prediction text
                if confidence >= self.withfiofnce_threshold:
                    text = f"Class: {class_name}"
                    color = (0, 255, 0)  # Verof si withfianza alta
                else:
                    text = f"Class: {class_name} (?)"
                    color = (0, 165, 255)  # Naranja si withfianza baja
                
                cv2.putText(frame, text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Confidence
                conf_text = f"Confidence: {confidence*100:.1f}%"
                cv2.putText(frame, withf_text, (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # FPS
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Instrucciones
                cv2.putText(frame, "ESC: Exit | ESPACIO: Capturer", 
                           (20, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Barra of withfianza
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
                    print("\n Saliendo...")
                    break
                elif key == ord(' '):  # ESPACIO
                    capture_count += 1
                    filename = f"capture_{capture_count}_{class_name}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f" Capture saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n  Interrumpido by user")
        
        finally:
            # Liberar recursos
            cap.release()
            cv2.ofstroyAllWindows()
            print("\n Recursos liberados")


def main():
    """Función main"""
    import argparse
    
    parser = argparse.ArgumentParser(ofscription='Clasificación en tiempo real with webcam')
    parser.add_argument('--model', type=str, required=True,
                       help='Ruta al model (.pth)')
    parser.add_argument('--info', type=str, required=True,
                       help='Ruta al file info.json')
    parser.add_argument('--camera', type=int, offault=0,
                       help='Camera ID')
    parser.add_argument('--threshold', type=float, offault=0.5,
                       help='Umbral of withfianza')
    
    args = parser.parse_args()
    
    classifier = WebcamClassifier(
        moofl_path=args.model,
        moofl_info_path=args.info,
        withfiofnce_threshold=args.threshold
    )
    
    classifier.run(camera_id=args.camera)


if __name__ == "__main__":
    # Ejemplo of uso
    if len(sys.argv) == 1:
        print(" Uso:")
        print("   python webcam_classifier.py --model models/saved/resnet_best.pth --info models/saved/resnet_info.json")
        print("\n   Primero ofbes entrenar un model with train_classifier.py")
    else:
        main()
