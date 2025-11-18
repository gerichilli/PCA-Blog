# predict.py
import argparse
import os
import warnings
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from classification import train_model 

LABEL_TO_NAME = {0: "Green", 1: "Red"}

def prepare_image(path):
    img = Image.open(path).convert("RGB").resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten().reshape(1, -1)

def predict_image(path, pca, scaler, feature_scaler, model):
    img_vec = prepare_image(path)

    img_scaled = scaler.transform(img_vec).astype(np.float64, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            img_pca = pca.transform(img_scaled)
    img_pca_scaled = feature_scaler.transform(img_pca)

    pred = model.predict(img_pca_scaled)[0]
    return LABEL_TO_NAME.get(pred, "Unknown"), pred

def _infer_label_from_name(filename):
    name = filename.lower()
    if "green" in name:
        return 0
    if "red" in name:
        return 1
    return None

def predict_folder(folder_path, pca, scaler, feature_scaler, model, infer_ground_truth=False):
    print(f"Testing folder: {folder_path}")
    y_true, y_pred = [], []

    for file in sorted(os.listdir(folder_path)):
        if file.startswith("."):
            continue

        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        path = os.path.join(folder_path, file)
        label_name, label_idx = predict_image(path, pca, scaler, feature_scaler, model)
        print(f"{file} ➝ {label_name}")

        if infer_ground_truth:
            inferred = _infer_label_from_name(file)
            if inferred is not None:
                y_true.append(inferred)
                y_pred.append(label_idx)

    accuracy = None
    if infer_ground_truth and y_true:
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Inferred folder accuracy: {accuracy:.4f} ({len(y_true)} labeled samples)")

    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Train apple classifier and test on a folder of images.")
    parser.add_argument(
        "--folder",
        default="dataset/test_apples",
        help="Folder containing images to classify (default: dataset/test_apples)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=100,
        help="Number of PCA components to retain (default: 100)",
    )
    parser.add_argument(
        "--infer-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Infer ground-truth labels from filenames that include 'green' or 'red'.",
    )
    args = parser.parse_args()

    model, pca, scaler, feature_scaler, metrics = train_model(
        n_components=args.n_components
    )
    print(f"Training metrics: {metrics}")

    infer_labels = args.infer_labels
    if infer_labels is None:
        infer_labels = os.path.abspath(args.folder) == os.path.abspath(
            "dataset/test_apples"
        )

    predict_folder(
        args.folder,
        pca,
        scaler,
        feature_scaler,
        model,
        infer_ground_truth=infer_labels,
    )

if __name__ == "__main__":
    main()
