🌀 Churn Prediction — End-to-End ML Project

A complete production-ready pipeline for customer churn prediction including data preprocessing, feature engineering, model training, evaluation, and API deployment using FastAPI.

🌐 Architecture Diagram

End-to-end ML Pipeline Overview:

                          ┌─────────────────────┐
                          │      Dataset         │
                          │     (Churn.csv)      │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Data Processing    │
                          │ - Cleaning           │
                          │ - Encoding (OHE)     │
                          │ - Numeric Scaling    │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  ML Pipeline (Sklearn)│
                          │  - ColumnTransformer  │
                          │  - RandomForest (80 trees)
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Trained Model      │
                          │    (model.pkl)       │
                          │    622 KB (Light)    │
                          └──────────┬───────────┘
                                     │
               ┌─────────────────────▼─────────────────────┐
               │          FastAPI Inference API            │
               │  / → health check                         │
               │  /predict → churn prediction              │
               └───────────────────────────────────────────┘
📁 churn-prediction-mlops/

├── 📂 api/
│   └── app.py                 # FastAPI server for inference

├── 📂 data/
│   └── Churn.csv


├── 📂 models/
│   └── model.pkl              # Light ML model (622 KB)

├── 📂 notebook/
│   └── EDA.ipynb              # Exploratory Data Analysis


├── 📂 src/
│   └── train.py               # Model training pipeline

└── 📄 README.md



📊 Model Pipeline (Sklearn)

The entire process is automated using a unified pipeline:

🔹 Preprocessing (ColumnTransformer)

OneHotEncoder for categorical features

Pass-through for numerical features

🔹 Model

A lightweight RandomForest model:

RandomForestClassifier(
    n_estimators=80,
    max_depth=10,
    random_state=42
)

🔹 Benefits

No manual encoding needed during inference

Clean API → only raw JSON required

Small model size → 622 KB

Fast inference

📈 Model Evaluation
✔ Accuracy

~80–82%

✔ Key Metrics (Typical for Telecom Churn dataset)
Metric	Value
Accuracy	~0.80
Precision	~0.77
Recall	~0.70
F1-score	~0.73

(These values are approximate; retraining may produce slightly different results.)



🔳 Confusion Matrix

Approximately:

TN: 1300

FP: 250

FN: 350

TP: 450

⚙️ Training the Model

Run:

python src/train.py


This will:

Load & clean data

Build preprocessing pipeline

Train RandomForest model

Save model.pkl into /models/

⚡ FastAPI Inference Server
✔ Start API locally:
uvicorn api.app:app --reload


Available endpoints:

🟢 GET /

Health check

{"message": "Churn Prediction API is running."}

🔮 POST /predict

Send customer profile → get churn prediction.

Request:

{
  "customerID": "0001-AA",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Credit card (automatic)",
  "MonthlyCharges": 79.85,
  "TotalCharges": 999.50
}


Response:

{
  "prediction": 0,
  "churn_probability": 0.395
}


🧾 Model Card
✔ Intended Use

Telecom churn prediction for customer retention analysis.

✔ Limitations

Does not model time-series behavior

One-hot encoding increases feature dimensionality

Bias risk from imbalanced data

✔ Ethical Considerations

Predictions should not be used for individual penalties, only for business insights.

🧭 Future Improvements

Add SHAP explainability

Add MLflow model tracking

Replace RandomForest with XGBoost / LightGBM

Add automated CI/CD for deployment

Monitor API latency & accuracy drift

👩‍💻 Author

Maryam Nasourinia
MSc Computational Sciences
Data Analyst • Machine Learning Engineer
