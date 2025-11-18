# dataset_loader.py
import os
import numpy as np # Handle matric, flatten
from PIL import Image # Open images, turn to grayscale, resize

def load_images(folder, label):
    """
    1. Load images from a given folder
    2. Convert them to grayscale,
    3. Resize them to 64×64
    4. Flatten them into 1D vectors (4096 dimensions),
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
        Array of shape (N, 4096), where N is the number of images.
        Each row represents one flattened grayscale image.
    labels : numpy.ndarray
        Array of shape (N,), containing the label for each image.
    """
    
    images, labels = [], []

    # Iterate through all files in the folder
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        # Open the image, convert to grayscale, and resize to 64×64
        img = Image.open(path).convert("L").resize((64, 64))

        # Convert to NumPy array and flatten to a 4096-dimensional vector
        images.append(np.array(img).flatten())

        # Append the label for this image
        labels.append(label)

    # Return as NumPy arrays for compatibility with ML models
    return np.array(images), np.array(labels)
