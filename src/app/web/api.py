"""
FastAPI Backend for Multimodal Classification
Endpoints for images, viofo and real-time webcam
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.image_classifier.resnet_classifier import ResNetClassifier
from src.models.image_classifier.mobilenet_classifier import MobileNetClassifier
from src.data.preprocessing.image_preprocessor import ImagePreprocessor

# Initialize FastAPI
app = FastAPI(
    title="Multimodal Classifier API",
    description="API for real-time image and video classification",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
PREPROCESSOR = None
CLASS_NAMES = []
MODEL_TYPE = "resnet"


def load_model(model_type: str = "resnet"):
    """Load pretrained model"""
    global MODEL, PREPROCESSOR, CLASS_NAMES, MODEL_TYPE
    
    MODEL_TYPE = model_type
    model_path = f"models/saved/{model_type}_best.pth"
    info_path = f"models/saved/{model_type}_info.json"
    
    print(f"Loading model {model_type}...")
    
    # Load model info
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    CLASS_NAMES = info['class_names']
    num_classes = info['num_classes']
    
    # Create model
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
    
    # Create preprocessor
    PREPROCESSOR = ImagePreprocessor(
        image_size=(224, 224),
        normalize=True,
        augmentation=False
    )
    
    print(f"Model {model_type} loaded successfully")
    print(f"   Classes: {CLASS_NAMES}")


def predict_image(image: np.ndarray) -> dict:
    """Perform prediction on an image"""
    global MODEL, PREPROCESSOR, CLASS_NAMES
    
    if MODEL is None:
        load_model()
    
    # Preprocess (returns PyTorch tensor thanks to ToTensorV2)
    image_tensor = PREPROCESSOR.preprocess(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = MODEL.predict(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_value = confidence.item() * 100
    
    # Get top 3 predictions
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
    """Load model on startup"""
    load_model("resnet")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>API running. Go to /docs to see documentation.</h1>")


@app.get("/api/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_type": MODEL_TYPE,
        "classes": CLASS_NAMES
    }


@app.post("/api/predict/image")
async def predict_uploaded_image(file: UploadFile = File(...)):
    """
    Predict class of uploaded image
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Predict
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
    WebSocket for real-time webcam streaming
    Client sends frames in base64, server responds with predictions
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive frame in base64
            data = await websocket.receive_text()
            
            # Decode image
            img_data = base64.b64decode(data.split(',')[1] if ',' in data else data)
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image)
            
            # Predict
            result = predict_image(image_np)
            
            # Send result
            await websocket.send_json(result)
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.post("/api/model/switch")
async def switch_model(model_type: str):
    """
    Switch between ResNet and MobileNet
    """
    if model_type not in ['resnet', 'mobilenet']:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Model must be 'resnet' or 'mobilenet'"}
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
    print("Starting FastAPI server")
    print("=" * 70)
    print("Server: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("WebSocket: ws://localhost:8000/ws/webcam")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
