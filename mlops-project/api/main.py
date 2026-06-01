import sys
import os
import csv
import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure src directory is in the path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import ChurnPredictor

# Optional Evidently AI import
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

app = FastAPI(
    title="Telecom Customer Churn Serving & Monitoring API",
    description="Production-ready FastAPI app deploying a Scikit-Learn RandomForest model pipeline, with automated logging and Evidently AI data drift monitoring.",
    version="1.0.0"
)

# Pydantic schemas for request validation
class CustomerInput(BaseModel):
    tenure: int = Field(..., description="Number of months the customer has stayed", json_schema_extra={"example": 12})
    MonthlyCharges: float = Field(..., description="The amount charged to the customer monthly", json_schema_extra={"example": 70.5})
    TotalCharges: float = Field(..., description="The total amount charged to the customer", json_schema_extra={"example": 846.0})
    Contract: str = Field(..., description="The contract term of the customer (Month-to-month, One year, Two year)", json_schema_extra={"example": "Month-to-month"})
    InternetService: str = Field(..., description="Customer's internet service provider (DSL, Fiber optic, No)", json_schema_extra={"example": "Fiber optic"})
    TechSupport: str = Field(..., description="Whether the customer has tech support (Yes, No, No internet service)", json_schema_extra={"example": "No"})
    OnlineSecurity: str = Field(..., description="Whether the customer has online security (Yes, No, No internet service)", json_schema_extra={"example": "No"})
    PaperlessBilling: str = Field(..., description="Whether the customer has paperless billing (Yes, No)", json_schema_extra={"example": "Yes"})

class PredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="Binary churn prediction (1 = Churn, 0 = Stay)")
    churn_probability: float = Field(..., description="The model's probability score for churning")
    churn_risk_level: str = Field(..., description="Inferred risk severity: High (>=0.7), Medium (>=0.4), Low (<0.4)")
    attrition_warning: bool = Field(..., description="Warning flag activated if predicted to churn")

# Global predictor instance
predictor = None
LOG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "serving_log.csv"))

def get_predictor():
    global predictor
    if predictor is None:
        try:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "model.pkl"))
            predictor = ChurnPredictor(model_path=model_path)
        except Exception as e:
            raise RuntimeError(f"Could not load trained pipeline model: {str(e)}")
    return predictor

def log_serving_request(features: dict, prediction_result: dict):
    """
    Logs incoming predictions in real-time to data/serving_log.csv to simulate
    a production data capture pipeline. This data is used for drift monitoring.
    """
    try:
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        file_exists = os.path.exists(LOG_FILE_PATH)
        
        # Merge input features with output labels and timestamp
        log_row = {**features, **prediction_result}
        log_row["timestamp"] = datetime.datetime.now().isoformat()
        
        # Ensure 'target' represents the prediction label for Evidently to analyze prediction drift
        log_row["target"] = prediction_result["churn_prediction"]

        headers = list(log_row.keys())
        
        with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_row)
    except Exception as e:
        print(f"Error logging serving request: {e}")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Telecom Customer Churn Prediction Serving & Monitoring Service",
        "docs_url": "/docs",
        "health_url": "/health",
        "monitoring_url": "/monitor",
        "status": "active"
    }

@app.get("/health")
def health_check():
    try:
        get_predictor()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "degraded", "model_loaded": False, "error": str(e)}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerInput):
    try:
        pred_svc = get_predictor()
        
        # Convert Pydantic payload to dictionary
        customer_dict = payload.model_dump()
        
        # Predict using our wrapper class
        results = pred_svc.predict([customer_dict])
        prediction = results[0]
        
        # Asynchronously log serving request for Evidently drift analysis
        log_serving_request(customer_dict, prediction)
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/monitor")
def monitor_drift():
    """
    Compares real-time serving requests log against baseline test training data.
    Generates an HTML Data Drift report and returns a summary JSON.
    """
    if not EVIDENTLY_AVAILABLE:
        return {
            "status": "error",
            "message": "Evidently AI is not installed. Please install 'evidently' dependency."
        }

    # Define paths
    ref_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test.csv"))
    report_html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "drift_report.html"))

    if not os.path.exists(ref_data_path):
        raise HTTPException(
            status_code=404, 
            detail="Reference baseline test dataset not found. Please execute the DVC preparation pipeline."
        )

    if not os.path.exists(LOG_FILE_PATH):
        return {
            "status": "waiting",
            "message": "No production serving requests have been logged yet. Please call /predict endpoint first to generate data logs."
        }

    try:
        # Load reference data (trained data splits)
        ref_df = pd.read_csv(ref_data_path)
        
        # Load current serving logs
        curr_df = pd.read_csv(LOG_FILE_PATH)
        
        # Drop columns not present in reference (like timestamp, risks, etc.)
        # Align columns to exactly match features
        columns_to_keep = [col for col in ref_df.columns if col in curr_df.columns]
        
        # Ensure we have enough serving logs to test drift (e.g. min 5 records)
        if len(curr_df) < 5:
            return {
                "status": "collecting",
                "serving_records_logged": len(curr_df),
                "message": "Need at least 5 logged requests to run data drift analysis. Please query the API more."
            }

        ref_analysis = ref_df[columns_to_keep]
        curr_analysis = curr_df[columns_to_keep]

        # Construct and run Evidently Data Drift report
        data_drift_report = Report(metrics=[
            DataDriftPreset()
        ])
        
        data_drift_report.run(reference_data=ref_analysis, current_data=curr_analysis)
        data_drift_report.save_html(report_html_path)

        # Retrieve a dictionary summary of drift results to return in JSON
        report_json = data_drift_report.as_dict()
        
        # Safely extract drift summary metrics
        metrics_summary = report_json.get("metrics", [{}])[0].get("result", {})
        drifted_features = metrics_summary.get("number_of_drifted_columns", 0)
        total_features = metrics_summary.get("number_of_columns", 0)
        share_of_drifted = metrics_summary.get("share_of_drifted_columns", 0.0)

        return {
            "status": "success",
            "report_generated_at": datetime.datetime.now().isoformat(),
            "serving_records_analyzed": len(curr_df),
            "drift_detected": bool(share_of_drifted > 0.5), # Drift flagged if > 50% columns drifted
            "drift_summary": {
                "drifted_features_count": drifted_features,
                "total_features_count": total_features,
                "share_of_drifted_features": float(share_of_drifted)
            },
            "view_report_url": "/monitor/report"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compile drift report: {str(e)}")

@app.get("/monitor/report")
def get_monitor_report():
    report_html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "drift_report.html"))
    if not os.path.exists(report_html_path):
        raise HTTPException(
            status_code=404, 
            detail="Monitoring HTML report has not been compiled yet. Please hit the '/monitor' endpoint first."
        )
    return FileResponse(report_html_path)
