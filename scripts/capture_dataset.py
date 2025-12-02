"""
Script auxiliar para tomar fotos con la webcam
Útil para crear tu propio dataset rápidamente
"""

import cv2
import os
from pathlib import Path
import time

class DatasetCapture:
    """Capturar fotos para dataset con webcam"""
    
    def __init__(self, output_dir: str = "data/raw/images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def capture_class(self, class_name: str, num_images: int = 20):
        """
        Capturar imágenes para una clase
        
        Args:
            class_name: Nombre de la clase (ej: 'laptop', 'mouse')
            num_images: Número de imágenes a capturar
        """
        # Crear carpeta para la clase
        class_dir = self.output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Abrir cámara
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara")
            return
        
        print("\n" + "=" * 60)
        print(f"📸 CAPTURA DE FOTOS - Clase: {class_name}")
        print("=" * 60)
        print(f"📁 Guardando en: {class_dir}")
        print(f"🎯 Objetivo: {num_images} fotos")
        print("\n⌨️  Controles:")
        print("   • ESPACIO - Tomar foto")
        print("   • ESC - Salir")
        print("=" * 60)
        print("\n💡 Tips:")
        print("   • Varía el ángulo en cada foto")
        print("   • Cambia la distancia al objeto")
        print("   • Prueba con diferente iluminación")
        print("   • Incluye diferentes fondos")
        print("\n")
        
        captured = 0
        
        try:
            while captured < num_images:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error al leer frame")
                    break
                
                # Mostrar contador
                display_frame = frame.copy()
                text = f"Fotos: {captured}/{num_images}"
                cv2.putText(display_frame, text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.putText(display_frame, "ESPACIO: Capturar | ESC: Salir", 
                           (20, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(f"Captura - {class_name}", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # ESPACIO
                    # Guardar foto
                    filename = class_dir / f"{class_name}_{captured+1:03d}.jpg"
                    cv2.imwrite(str(filename), frame)
                    
                    captured += 1
                    print(f"✓ Foto {captured}/{num_images} guardada: {filename.name}")
                    
                    # Feedback visual
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), 
                                (flash.shape[1], flash.shape[0]), 
                                (255, 255, 255), -1)
                    cv2.imshow(f"Captura - {class_name}", flash)
                    cv2.waitKey(100)
                    
                elif key == 27:  # ESC
                    print("\n⚠️  Captura cancelada por usuario")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrumpido por usuario")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"\n✅ Captura completada: {captured} fotos de '{class_name}'")
        print(f"📁 Guardadas en: {class_dir}")
        
        return captured

def main():
    """Función principal interactiva"""
    print("=" * 60)
    print("📸 CAPTURADOR DE DATASET")
    print("=" * 60)
    
    capturer = DatasetCapture()
    
    # Obtener información del usuario
    print("\n¿Cuántas clases vas a capturar?")
    try:
        num_classes = int(input("Número de clases: "))
    except ValueError:
        print("❌ Entrada inválida")
        return
    
    print("\n¿Cuántas fotos por clase?")
    try:
        photos_per_class = int(input("Fotos por clase (recomendado: 15-25): "))
    except ValueError:
        print("❌ Entrada inválida")
        return
    
    # Capturar cada clase
    for i in range(num_classes):
        print(f"\n{'='*60}")
        print(f"CLASE {i+1}/{num_classes}")
        print("="*60)
        
        class_name = input(f"Nombre de la clase {i+1}: ").strip()
        
        if not class_name:
            print("⚠️  Nombre vacío, usando 'clase_{i+1}'")
            class_name = f"clase_{i+1}"
        
        # Limpiar nombre (remover caracteres no válidos)
        class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_'))
        class_name = class_name.replace(' ', '_')
        
        print(f"\n🎬 Prepárate para capturar '{class_name}'")
        print("   Posiciona el objeto frente a la cámara")
        input("   Presiona ENTER cuando estés listo...")
        
        capturer.capture_class(class_name, photos_per_class)
        
        if i < num_classes - 1:
            print("\n⏸️  Cambia de objeto para la siguiente clase")
            input("   Presiona ENTER para continuar...")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("✅ CAPTURA COMPLETADA")
    print("=" * 60)
    
    # Contar fotos
    data_dir = Path("data/raw/images")
    total_images = 0
    
    print("\n📊 Resumen del dataset:")
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            total_images += len(images)
            print(f"   • {class_dir.name}: {len(images)} imágenes")
    
    print(f"\n   Total: {total_images} imágenes")
    
    if total_images >= 50:
        print("\n✅ Dataset listo para entrenamiento!")
        print("\n🚀 Siguiente paso:")
        print("   python src/training/image/train_classifier.py")
    else:
        print(f"\n⚠️  Recomendación: al menos 50-70 imágenes")
        print(f"   Faltan ~{50-total_images} imágenes")

if __name__ == "__main__":
    main()
