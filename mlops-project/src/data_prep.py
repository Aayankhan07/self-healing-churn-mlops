import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def generate_synthetic_churn_data(size=5000, random_state=42):
    """
    Generates a highly realistic synthetic telecom customer churn dataset.
    Establishes real relationships between features and churn.
    """
    np.random.seed(random_state)

    # 1. Generate demographic & billing features
    tenure = np.random.randint(1, 73, size=size)  # 1 to 72 months
    
    # Contract: Month-to-month has higher propensity, One year medium, Two year low
    contract_probs = [0.55, 0.25, 0.20]
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=size, p=contract_probs)
    
    # InternetService: DSL, Fiber optic, No
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=size, p=[0.35, 0.45, 0.20])
    
    # TechSupport & OnlineSecurity (conditioned on having internet service)
    tech_support = []
    online_security = []
    for service in internet_service:
        if service == "No":
            tech_support.append("No internet service")
            online_security.append("No internet service")
        else:
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))
            online_security.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            
    tech_support = np.array(tech_support)
    online_security = np.array(online_security)
    
    paperless_billing = np.random.choice(["Yes", "No"], size=size, p=[0.6, 0.4])
    
    # Monthly Charges based on internet service
    monthly_charges = []
    for service in internet_service:
        if service == "Fiber optic":
            monthly_charges.append(np.random.uniform(70.0, 120.0))
        elif service == "DSL":
            monthly_charges.append(np.random.uniform(40.0, 80.0))
        else:
            monthly_charges.append(np.random.uniform(18.0, 30.0))
            
    monthly_charges = np.array(monthly_charges)
    total_charges = tenure * monthly_charges * np.random.uniform(0.95, 1.05, size=size)
    
    # 2. Build structured probability of churn (log odds logic)
    # Higher tenure = lower churn
    # Month-to-month = higher churn
    # Higher monthly charges = higher churn
    # No tech support / No online security = higher churn
    log_odds = (
        -1.5 
        - 0.04 * tenure 
        + 0.015 * monthly_charges 
        + 1.8 * (contract == "Month-to-month")
        - 0.5 * (contract == "Two year")
        + 0.8 * (tech_support == "No")
        + 0.6 * (online_security == "No")
        + np.random.normal(0, 0.5, size=size)  # adding noise
    )
    
    prob = 1 / (1 + np.exp(-log_odds))
    churn = (prob > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": np.round(monthly_charges, 2),
        "TotalCharges": np.round(total_charges, 2),
        "Contract": contract,
        "InternetService": internet_service,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
        "PaperlessBilling": paperless_billing,
        "target": churn  # 'target' represents Churn status
    })
    
    return df

def prepare_data():
    params = load_params()
    random_state = params["base"]["random_state"]
    test_size = params["data_prep"]["test_size"]

    # Ensure output directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 1. Generate or fetch data
    print("Generating synthetic Customer Churn dataset...")
    df = generate_synthetic_churn_data(size=5000, random_state=random_state)

    # Save raw dataset
    raw_path = os.path.join("data", "raw", "churn_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path} (Shape: {df.shape})")

    # 2. Split dataset into train/test sets
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df["target"]
    )

    # Save processed datasets
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved prepared train data (Shape: {train_df.shape}) to {train_path}")
    print(f"Saved prepared test data (Shape: {test_df.shape}) to {test_path}")

if __name__ == "__main__":
    prepare_data()
