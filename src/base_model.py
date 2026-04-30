import joblib
from pathlib import Path

class BaseModel:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path="models"):
        Path(path).mkdir(exist_ok=True)
        joblib.dump(self.model, f"{path}/{self.name}.pkl")

    def load(self, path="models"):
        self.model = joblib.load(f"{path}/{self.name}.pkl")