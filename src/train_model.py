import sys, os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 👇 Add project root (one level up) to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import load_and_clean  # now works fine

def train_model():
    X, y = load_and_clean()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    with mlflow.start_run():
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        # Log parameters & metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(clf, "model")

    print(f"Model trained with accuracy: {acc:.2f}")

if __name__ == "__main__":
    train_model()
