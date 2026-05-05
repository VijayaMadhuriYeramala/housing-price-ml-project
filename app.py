from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

@app.get("/")
def home():
    return {"message": "Housing price ML API is running"}

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)

    return {"predicted_house_value": float(prediction[0])}
