from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/attrition_model.pkl")

@app.get("/")
def home():
    return {"message": "HR Analytics API Running"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}