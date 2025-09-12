import pandas as pd

def load_and_clean(path="data/train.csv"):
    df = pd.read_csv(path)
    # Fill missing ages with median
    df["Age"].fillna(df["Age"].median(), inplace=True)
    # Encode Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    # Select features
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    y = df["Survived"]
    return X, y
