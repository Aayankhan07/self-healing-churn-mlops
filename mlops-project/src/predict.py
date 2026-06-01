import os
import joblib
import pandas as pd
import numpy as np

class ChurnPredictor:
    def __init__(self, model_path="data/model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained pipeline model not found at {model_path}. "
                "Ensure that the DVC pipeline or training script has run."
            )
        print(f"Loading churn prediction pipeline from {model_path}...")
        self.pipeline = joblib.load(model_path)
        
        # Define the exact columns that the training preprocessing pipeline expects
        self.feature_columns = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "InternetService",
            "TechSupport",
            "OnlineSecurity",
            "PaperlessBilling"
        ]

    def predict(self, customer_data: list):
        """
        Predict churn for customer entries.
        
        :param customer_data: A list of dictionaries representing customers, e.g.:
            [
                {
                    "tenure": 12,
                    "MonthlyCharges": 70.5,
                    "TotalCharges": 846.0,
                    "Contract": "Month-to-month",
                    "InternetService": "Fiber optic",
                    "TechSupport": "No",
                    "OnlineSecurity": "No",
                    "PaperlessBilling": "Yes"
                }
            ]
        :return: A list of dictionaries with predictions, probabilities, and warnings.
        """
        # Convert list of dicts to DataFrame with correct column ordering
        df = pd.DataFrame(customer_data)
        
        # Ensure all columns exist and fill missing ones with logical defaults if necessary
        for col in self.feature_columns:
            if col not in df.columns:
                if col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                    df[col] = 0.0
                else:
                    df[col] = "No"
                    
        # Order columns to match training schema
        df = df[self.feature_columns]

        # Predict classes and probabilities
        classes = self.pipeline.predict(df)
        probabilities = self.pipeline.predict_proba(df)[:, 1]

        results = []
        for i, (pred_class, prob) in enumerate(zip(classes, probabilities)):
            results.append({
                "churn_prediction": int(pred_class),
                "churn_probability": float(prob),
                "churn_risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low",
                "attrition_warning": bool(pred_class == 1)
            })

        return results

if __name__ == "__main__":
    # Example local CLI prediction simulation
    try:
        predictor = ChurnPredictor()
        
        # High churn risk profile (Month-to-month, High monthly charges, no security)
        high_risk_customer = [{
            "tenure": 3,
            "MonthlyCharges": 105.4,
            "TotalCharges": 316.2,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "PaperlessBilling": "Yes"
        }]

        # Low churn risk profile (Two year contract, DSL, has TechSupport/Security)
        low_risk_customer = [{
            "tenure": 60,
            "MonthlyCharges": 45.0,
            "TotalCharges": 2700.0,
            "Contract": "Two year",
            "InternetService": "DSL",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "PaperlessBilling": "No"
        }]

        print("\n--- Testing High Risk Customer ---")
        print(predictor.predict(high_risk_customer))

        print("\n--- Testing Low Risk Customer ---")
        print(predictor.predict(low_risk_customer))
        
    except Exception as e:
        print(f"CLI Prediction test not run (perhaps model is not trained yet): {e}")
        print("Please run data preparation and model training stages first.")
