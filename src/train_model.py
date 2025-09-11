import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv('data/train.csv')

# Preprocess
data['Age'].fillna(data['Age'].mean(), inplace=True)
X = data[['Pclass','Sex','Age','SibSp','Parch','Fare']].copy()
X['Sex'] = X['Sex'].map({'male':0,'female':1})
X = X.apply(pd.to_numeric, errors='coerce')

y = data['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'titanic_model.pkl')
