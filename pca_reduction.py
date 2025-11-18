# pca_reduction.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataset_loader import load_images

def run_pca():
    """
    Load green_apple and red_apple images, combine them into a single dataset,
    apply PCA to reduce dimensionality from 4096 to 100 components,
    and return the transformed data along with their labels and the PCA model.

    Returns
    -------
    X_pca : numpy.ndarray
        PCA-transformed feature matrix.
    y : numpy.ndarray
        Label array for all images (0 = green_apple, 1 = red_apple).
    pca : sklearn.decomposition.PCA
        The trained PCA transformer.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler from training step
    """

    # Load preprocessed green_apple images (label = 0)
    X_green_apples, y_green_apples = load_images("dataset/green_apples", label=0) # Images of green_apple
    # Load preprocessed red_apple images (label = 1)
    X_red_apples, y_red_apples = load_images("dataset/red_apples", label=1) # Images of red_apple

    # Combine green_apple + red_apple feature matrices into one dataset
    X = np.vstack([X_green_apples, X_red_apples])
    y = np.hstack([y_green_apples, y_red_apples])

    # Scale before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create PCA with 100 components
    pca = PCA(n_components=100, random_state=42)

    # Fit PCA on X and transform it
    X_pca = pca.fit_transform(X_scaled)

    print("Original shape:", X.shape)    # (6472, 4096)
    print("After PCA:", X_pca.shape)     # (6472, 100)

    return X_pca, y, pca, scaler


run_pca()