import os
import pandas as pd

def load_and_clean():
    # Get absolute path to data/train.csv
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "train.csv")

    df = pd.read_csv(data_path)

    # Minimal preprocessing
    df = df.dropna(subset=["Age", "Fare", "Embarked"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    y = df["Survived"]

    return X, y
