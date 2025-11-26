from fastapi import FastAPI
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/model.pkl")   # models/model.pkl

app = FastAPI(title="Churn Prediction API")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running."}

@app.post("/predict")
def predict(data: dict):

    # Convert the incoming JSON to DataFrame
    df = pd.DataFrame([data])

    # Ensure column order matches training
    df = df.reindex(columns=[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ], fill_value=0)

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }

