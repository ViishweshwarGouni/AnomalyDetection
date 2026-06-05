from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
import joblib
import os

os.makedirs("models", exist_ok=True)

iris = load_iris()
X = iris.data

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X)

joblib.dump(
    model,
    "models/isolation_forest.pkl"
)

print("✅ Isolation Forest model saved successfully!")