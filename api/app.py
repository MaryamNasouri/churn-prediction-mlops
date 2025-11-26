from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

app = FastAPI(title="Churn Prediction API")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running."}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

       
        if "customerID" not in df.columns:
            df["customerID"] = "0000-XYZ"

        expected_cols = [
            'customerID',   
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]

        df = df.reindex(columns=expected_cols)

        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "churn_probability": float(prob)
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "details": traceback.format_exc()}
