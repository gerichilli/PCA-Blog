# dataset_loader.py
import os
import numpy as np # Handle matric, flatten
from PIL import Image # Open images, convert to RGB, resize

def load_images(folder, label):
    """
    1. Load images from a given folder
    2. Convert them to RGB (giữ nguyên thông tin màu),
    3. Resize them to 64×64
    4. Normalize to [0, 1] and flatten to 1D vectors (64×64×3 = 12,288 dimensions),
    5. Return both the processed images and their corresponding labels (0 for green_apple, 1 for red_apple) red_apple).

    Parameters
    ----------
    folder : str
        Path to the folder containing the images.
    label : int
        The numerical label assigned to all images in this folder 
        (0 for green_apple, 1 for red_apple).

    Returns
    -------
    images : numpy.ndarray
        Array of shape (N, 12288), where N is the number of images.
        Each row represents one flattened RGB image.
    labels : numpy.ndarray
        Array of shape (N,), containing the label for each image.
    """
    
    images, labels = [], []

    # Iterate through all files in the folder
    for filename in os.listdir(folder):
        if filename.startswith("."):
            continue

        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        path = os.path.join(folder, filename)

        # Open the image, convert to RGB, and resize to 64×64
        img = Image.open(path).convert("RGB").resize((64, 64))

        # Convert to float32 NumPy array in [0, 1] and flatten
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr.flatten())

        # Append the label for this image
        labels.append(label)

    # Return as NumPy arrays for compatibility with ML models
    return np.array(images), np.array(labels)
