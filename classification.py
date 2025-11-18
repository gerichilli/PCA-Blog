# classification.py
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from pca_reduction import run_pca

np.seterr(divide="ignore", invalid="ignore", over="ignore")

def train_model(test_size=0.2, random_state=42, n_components=100):
    """
    Train a logistic regression model on PCA reduced features.

    Parameters
    ----------
    test_size : float
        Fraction of the dataset to hold out for evaluation.
    random_state : int
        Seed for reproducibility.
    n_components : int
        Number of PCA components to retain.

    Returns
    -------
    model : sklearn.linear_model.LogisticRegressionCV
        Trained classifier.
    pca : sklearn.decomposition.PCA
        Fitted PCA transformer.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler.
    feature_scaler : sklearn.preprocessing.StandardScaler
        Scaler applied after PCA so inference matches training.
    metrics : dict
        Dictionary containing evaluation metadata.
    """

    X_pca, y, pca, scaler, feature_scaler = run_pca(n_components=n_components)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Cross-validated logistic regression improves generalization accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    model = LogisticRegressionCV(
        Cs=np.logspace(-2, 2, 9),
        cv=cv,
        max_iter=2000,
        scoring="accuracy",
        n_jobs=1,
        refit=True,
        solver="liblinear",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    metrics = {
        "test_accuracy": acc,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_C": float(model.C_[0]),
    }

    print(
        f"Test accuracy: {acc:.4f} "
        f"(train={len(X_train)} samples, test={len(X_test)} samples, best C={model.C_[0]:.4f})"
    )

    return model, pca, scaler, feature_scaler, metrics
