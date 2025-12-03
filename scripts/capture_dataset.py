"""
Script auxiliar for tomar photos with la webcam
Útil for crear tu propio dataset quickmente
"""

import cv2
import os
from pathlib import Path
import time

class DatasetCapture:
    """Capturer photos for dataset with webcam"""
    
    def __init__(self, output_dir: str = "data/raw/images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def capture_class(self, class_name: str, num_images: int = 20):
        """
        Capturer images for una clase
        
        Args:
            class_name: Name of la clase (ej: 'laptop', 'mouse')
            num_images: Number of images a capturer
        """
        # Create carpeta for la clase
        class_dir = self.output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Open camera
        cap = cv2.ViofoCapture(0)
        
        if not cap.isOpened():
            print(" Error: Could not open camera")
            return
        
        print("\n" + "=" * 60)
        print(f" CAPTURA DE FOTOS - Class: {class_name}")
        print("=" * 60)
        print(f" Saving en: {class_dir}")
        print(f" Target: {num_images} photos")
        print("\n⌨  Controles:")
        print("   • ESPACIO - Take photo")
        print("   • ESC - Exit")
        print("=" * 60)
        print("\n Tips:")
        print("   • Varía el ángulo en each photo")
        print("   • Cambia la distancia al object")
        print("   • Prueba with diferente iluminación")
        print("   • Incluye diferentes fondos")
        print("\n")
        
        captured = 0
        
        try:
            while captured < num_images:
                ret, frame = cap.read()
                
                if not ret:
                    print(" Error al leer frame")
                    break
                
                # Mostrar withtador
                display_frame = frame.copy()
                text = f"Photos: {captured}/{num_images}"
                cv2.putText(display_frame, text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(display_frame, "ESPACIO: Capturer | ESC: Exit", 
                           (20, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(f"Capture - {class_name}", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # ESPACIO
                    # Save photo
                    filename = class_dir / f"{class_name}_{captured+1:03d}.jpg"
                    cv2.imwrite(str(filename), frame)
                    
                    captured += 1
                    print(f" Photo {captured}/{num_images} saved: {filename.name}")
                    
                    # Feedback visual
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), 
                                (flash.shape[1], flash.shape[0]), 
                                (255, 255, 255), -1)
                    cv2.imshow(f"Capture - {class_name}", flash)
                    cv2.waitKey(100)
                    
                elif key == 27:  # ESC
                    print("\n  Capture cancelled by user")
                    break
        
        except KeyboardInterrupt:
            print("\n  Interrumpido by user")
        
        finally:
            cap.release()
            cv2.ofstroyAllWindows()
        
        print(f"\n Capture completed: {captured} photos of '{class_name}'")
        print(f" Saveds en: {class_dir}")
        
        return captured

def main():
    """Función main interactiva"""
    print("=" * 60)
    print(" CAPTURADOR DE DATASET")
    print("=" * 60)
    
    capturer = DatasetCapture()
    
    # Get información of user
    print("\nHow many classes will you capturer?")
    try:
        num_classes = int(input("Number of classes: "))
    except ValueError:
        print(" Entrada inválida")
        return
    
    print("\nHow many photos by clase?")
    try:
        photos_per_class = int(input("Photos by clase (recommended: 15-25): "))
    except ValueError:
        print(" Entrada inválida")
        return
    
    # Capturer each clase
    for i in range(num_classes):
        print(f"\n{'='*60}")
        print(f"CLASE {i+1}/{num_classes}")
        print("="*60)
        
        class_name = input(f"Name of la clase {i+1}: ").strip()
        
        if not class_name:
            print("  Name vacío, usando 'clase_{i+1}'")
            class_name = f"clase_{i+1}"
        
        # Clear nombre (remosee caracteres no válidos)
        class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_'))
        class_name = class_name.replace(' ', '_')
        
        print(f"\n Get ready for capturer '{class_name}'")
        print("   Position the object in front of the camera")
        input("   Presiona ENTER cuando estés listo...")
        
        capturer.capture_class(class_name, photos_per_class)
        
        if i < num_classes - 1:
            print("\n⏸  Cambia of object for la siguiente clase")
            input("   Presiona ENTER for withtinuar...")
    
    # Resumen final
    print("\n" + "=" * 60)
    print(" CAPTURA COMPLETADA")
    print("=" * 60)
    
    # Count photos
    data_dir = Path("data/raw/images")
    total_images = 0
    
    print("\n Resumen of dataset:")
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            total_images += len(images)
            print(f"   • {class_dir.name}: {len(images)} images")
    
    print(f"\n   Total: {total_images} images")
    
    if total_images >= 50:
        print("\n Dataset listo for training!")
        print("\n Siguiente paso:")
        print("   python src/training/image/train_classifier.py")
    else:
        print(f"\n  Recomendación: al menos 50-70 images")
        print(f"   Faltan ~{50-total_images} images")

if __name__ == "__main__":
    main()
