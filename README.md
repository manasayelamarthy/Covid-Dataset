# COVID-19 Classification using Deep Learning

## About the Project
This project is a deep learning-based image classification model designed to detect COVID-19, normal, and viral pneumonia cases from chest X-ray images. It uses multiple models, including ResNet, EfficientNet, and a custom CNN. The best-performing model, ResNet, achieved 98% accuracy.

## Deployed Application
The frontend is built using Vercel v0 AI and deployed at:

[**Live Demo**]([YOUR_VERCEL_DEPLOYMENT_URL](https://v0-react-js-image-predictor.vercel.app/))

### Screenshot of the Webpage


## Dataset Details
The dataset consists of three classes:
- COVID-19
- Normal
- Viral Pneumonia

The dataset was sourced from Kaggle. You can find it here:
[Kaggle Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

## Model Details
This project utilizes three different deep learning models:
- **ResNet** (98% accuracy)
- **EfficientNet**
- **Custom CNN model**
```
The training scripts are designed to be scalable, with automatic configuration and log saving.
```
## Deployment Details
The project is deployed using:
- **Backend:** FastAPI, Docker, and Google Cloud Run
- **Frontend:** Vercel v0 AI

## Instructions to Reproduce
To train and run the model on your local machine:

1. Clone the repository:
   ```sh
   git clone YOUR_REPO_URL
   cd covid-classification
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Train the model using the training script:
   ```sh
   python train.py --model resnet --epochs 10 --batch_size 32
   ```
   The script supports various arguments for customization.

4. To run the API:
   ```sh
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

Let me know if you have any questions or need help setting things up.

