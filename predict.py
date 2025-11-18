# predict.py
import os
import numpy as np
from PIL import Image
from classification import train_model 

def prepare_image(path):
    img = Image.open(path).convert("L").resize((64, 64))
    return np.array(img).flatten().reshape(1, -1)

def predict_image(path, pca, scaler, model):
    img_vec = prepare_image(path)

    # Apply the same scaling used during training
    img_scaled = scaler.transform(img_vec)

    # Apply PCA transform
    img_pca = pca.transform(img_scaled)

    # Predict label
    pred = model.predict(img_pca)[0]
    return "Green" if pred == 0 else "Red"

def predict_folder(folder_path, pca, scaler, model):
    print(f"Testing folder: {folder_path}")
    
    for file in os.listdir(folder_path):
        # Skip hidden/system files such as .DS_Store
        if file.startswith("."):
            continue

        # Only process common image file types
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        path = os.path.join(folder_path, file)
        label = predict_image(path, pca, scaler, model)
        print(f"{file} ➝ {label}")

model, pca, scaler = train_model()

# Test on folders
predict_folder("dataset/test_apples", pca, scaler, model)