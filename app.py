from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import base64
import io
import os
import logging
import time
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# Crear carpeta 'upload' si no existe
UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clases del modelo (Skin Cancer)
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Seleccionar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

# Cargar modelo ResNet101
model = models.resnet101(weights=None)  # No usar pesos preentrenados por seguridad
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # Ajustar capa de salida
model.load_state_dict(torch.load("./best_model.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def decode_base64_image(image_base64: str) -> str:
    """Decodifica imagen base64, guarda en carpeta upload y retorna la ruta"""
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        timestamp = int(time.time())
        filename = f"uploaded_{timestamp}.jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(path, format="JPEG")
        return path
    except Exception as e:
        logger.error(f"Error al decodificar imagen: {str(e)}")
        raise HTTPException(status_code=400, detail="Imagen base64 inválida")

@app.get("/test")
def test():
    return {"message": "API funcionando correctamente"}

@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}

@app.post("/predict/")
async def predict_image_base64(data: dict):
    """Recibe imagen en base64, predice clase de cáncer de piel, retorna clase y confianza"""
    if "image" not in data:
        raise HTTPException(status_code=400, detail="No se proporcionó imagen en la solicitud")

    try:
        # Guardar imagen
        image_path = decode_base64_image(data["image"])

        # Preprocesamiento
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predicción
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)
            predicted_class = CLASSES[predicted_class_idx.item()]

        # Eliminar imagen después de usarla
        os.remove(image_path)

        return {
            "prediction": predicted_class,
            "confidence": float(confidence.item())
        }
    except Exception as e:
        logger.error(f"Error durante predicción: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en la predicción")
