from fastapi import FastAPI, File, UploadFile
import torch
import numpy as np
import cv2
from training_scripts import Tester

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model tester
tester = Tester(model="resnet", checkpoint_path="checkpoint/resnet.pth", device=device)

app = FastAPI()

@app.get("/")
def root():
    return {"response": "Covid Disease Prediction"}

@app.post("/predict/")
async def predict(files: list[UploadFile] = File(...)):
    """Endpoint to receive images and return predictions"""
    images = []
    for file in files:
        image_bytes = await file.read()
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        images.append(image)
    
    labels, confidences = tester.predict(images)

    response = {'predictions' : []}

    for i, label in enumerate(labels):
        response['predictions'].append(
            {'label': label,
             'confidence': confidences[i]}
        )
    
    return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)