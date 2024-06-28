from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np

model = torch.load('chemin/vers/le/modele.pt')
model.eval()

app = FastAPI()

def predict(image):
    # Prétraitement de l'image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)
    
    # Prédiction
    with torch.no_grad():
        output = model(image)
    
    return torch.argmax(output, dim=1).item()

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction MNIST"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)