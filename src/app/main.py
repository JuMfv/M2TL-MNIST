# src/app/main.py

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from src.model.model import Net  # Assurez-vous que ce chemin d'importation est correct

app = FastAPI()

# Charger le modèle
model = Net()
model.load_state_dict(torch.load("model/mnist_cnn.pt"))
model.eval()

# Définir la transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Prétraiter l'image
    image_tensor = transform(image).unsqueeze(0)
    
    # Faire la prédiction
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
    
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
