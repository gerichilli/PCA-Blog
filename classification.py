# classification.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pca_reduction import run_pca

def train_model():
    X_pca, y, pca, scaler = run_pca()

    # TRAINING
    # Split the PCA features and labels into train and test sets
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    # Initialize Logistic Regression classifier
    # max_iter is set high enough to make sure the model converges
    model = LogisticRegression(max_iter=2000)

    # Train the model on the PCA-transformed training data
    model.fit(X_train, y_train)

    # Predict labels for the test set
    y_pred = model.predict(X_test)

    # Compute classification accuracy
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc) # Accuracy: 0.6640926640926641

    return model, pca, scaler

