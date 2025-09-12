import mlflow.pyfunc
import pandas as pd

def predict(model_uri, input_data):
    model = mlflow.pyfunc.load_model(model_uri)
    df = pd.DataFrame(input_data)
    preds = model.predict(df)
    return preds

if __name__ == "__main__":
    sample = {"Pclass": [3], "Sex": [0], "Age": [22], "SibSp": [1], "Parch": [0], "Fare": [7.25]}
    print(predict("models:/titanic_model/Production", sample))
