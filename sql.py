# Save this as train_model.py and run it once
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle

# Load dataset
# iris = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\CSI\Iris.csv")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
