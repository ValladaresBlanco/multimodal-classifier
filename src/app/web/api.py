"""
FastAPI Backend para Clasificación Multimodal
Endpoints para imágenes, video y webcam en tiempo real
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import io
import json
import base64
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.image_classifier.resnet_classifier import ResNetClassifier
from src.models.image_classifier.mobilenet_classifier import MobileNetClassifier
from src.data.preprocessing.image_preprocessor import ImagePreprocessor

# Inicializar FastAPI
app = FastAPI(
    title="Clasificador Multimodal API",
    description="API para clasificación de imágenes y videos en tiempo real",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
MODEL = None
PREPROCESSOR = None
CLASS_NAMES = []
MODEL_TYPE = "resnet"


def load_model(model_type: str = "resnet"):
    """Cargar modelo preentrenado"""
    global MODEL, PREPROCESSOR, CLASS_NAMES, MODEL_TYPE
    
    MODEL_TYPE = model_type
    model_path = f"models/saved/{model_type}_best.pth"
    info_path = f"models/saved/{model_type}_info.json"
    
    print(f"🔄 Cargando modelo {model_type}...")
    
    # Cargar info del modelo
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    CLASS_NAMES = info['class_names']
    num_classes = info['num_classes']
    
    # Crear modelo
    config = {
        'num_classes': num_classes,
        'pretrained': False,
        'freeze_backbone': False
    }
    
    if model_type == 'resnet':
        MODEL = ResNetClassifier(config)
    else:
        MODEL = MobileNetClassifier(config)
    
    MODEL.build_model()
    MODEL.load_model(model_path)
    MODEL.model.eval()
    
    # Crear preprocessor
    PREPROCESSOR = ImagePreprocessor(
        image_size=(224, 224),
        normalize=True,
        augmentation=False
    )
    
    print(f"✅ Modelo {model_type} cargado correctamente")
    print(f"   Clases: {CLASS_NAMES}")


def predict_image(image: np.ndarray) -> dict:
    """Realizar predicción en una imagen"""
    global MODEL, PREPROCESSOR, CLASS_NAMES
    
    if MODEL is None:
        load_model()
    
    # Preprocesar (retorna tensor PyTorch gracias a ToTensorV2)
    image_tensor = PREPROCESSOR.preprocess(image)
    
    # Añadir batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Predecir
    with torch.no_grad():
        outputs = MODEL.predict(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_value = confidence.item() * 100
    
    # Obtener top 3 predicciones
    top3_prob, top3_idx = torch.topk(probabilities, min(3, len(CLASS_NAMES)))
    top3_predictions = [
        {
            "class": CLASS_NAMES[idx.item()],
            "confidence": prob.item() * 100
        }
        for prob, idx in zip(top3_prob[0], top3_idx[0])
    ]
    
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence_value, 2),
        "top3_predictions": top3_predictions,
        "all_classes": CLASS_NAMES
    }


@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar"""
    load_model("resnet")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Servir página principal"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>API funcionando. Accede a /docs para ver la documentación.</h1>")


@app.get("/api/health")
async def health_check():
    """Verificar estado de la API"""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_type": MODEL_TYPE,
        "classes": CLASS_NAMES
    }


@app.post("/api/predict/image")
async def predict_uploaded_image(file: UploadFile = File(...)):
    """
    Predecir clase de imagen subida
    """
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convertir a numpy array
        image_np = np.array(image)
        
        # Predecir
        result = predict_image(image_np)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            **result
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.websocket("/ws/webcam")
async def websocket_webcam(websocket: WebSocket):
    """
    WebSocket para streaming de webcam en tiempo real
    Cliente envía frames en base64, servidor responde con predicciones
    """
    await websocket.accept()
    
    try:
        while True:
            # Recibir frame en base64
            data = await websocket.receive_text()
            
            # Decodificar imagen
            img_data = base64.b64decode(data.split(',')[1] if ',' in data else data)
            image = Image.open(io.BytesIO(img_data))
            
            # Convertir a RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image)
            
            # Predecir
            result = predict_image(image_np)
            
            # Enviar resultado
            await websocket.send_json(result)
    
    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        print(f"Error en WebSocket: {e}")
        await websocket.close()


@app.post("/api/model/switch")
async def switch_model(model_type: str):
    """
    Cambiar entre ResNet y MobileNet
    """
    if model_type not in ['resnet', 'mobilenet']:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Modelo debe ser 'resnet' o 'mobilenet'"}
        )
    
    try:
        load_model(model_type)
        return JSONResponse(content={
            "success": True,
            "model": model_type,
            "classes": CLASS_NAMES
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 Iniciando servidor FastAPI")
    print("=" * 70)
    print("📡 Servidor: http://localhost:8000")
    print("📚 Documentación: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws/webcam")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
