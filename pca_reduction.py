# pca_reduction.py
import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataset_loader import load_images

def run_pca(n_components=100):
    """
    Load green_apple and red_apple images, combine them into a single RGB dataset,
    apply PCA to reduce dimensionality from 12,288 to configurable components,
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
        Fitted scaler from training step.
    feature_scaler : sklearn.preprocessing.StandardScaler
        Scaler applied after PCA to keep features well conditioned.
    """

    # Load preprocessed green_apple images (label = 0)
    X_green_apples, y_green_apples = load_images("dataset/green_apples", label=0) # Images of green_apple
    # Load preprocessed red_apple images (label = 1)
    X_red_apples, y_red_apples = load_images("dataset/red_apples", label=1) # Images of red_apple

    # Combine green_apple + red_apple feature matrices into one dataset
    X = np.vstack([X_green_apples, X_red_apples])
    y = np.hstack([y_green_apples, y_red_apples])

    # Scale before PCA
    scaler = StandardScaler(with_mean=True, with_std=False)
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.astype(np.float64, copy=False)

    # Create PCA with configurable number of components
    pca = PCA(
        n_components=n_components,
        random_state=42,
        svd_solver="full",
        whiten=False,
    )

    # Fit PCA on X and transform it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            X_pca = pca.fit_transform(X_scaled)

    # Scale PCA features to keep them well conditioned for linear models
    feature_scaler = StandardScaler()
    X_pca_scaled = feature_scaler.fit_transform(X_pca)

    print("Original shape:", X.shape)
    print("After PCA:", X_pca.shape)

    return X_pca_scaled, y, pca, scaler, feature_scaler
