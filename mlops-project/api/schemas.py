"""
Pydantic v2 input/output schemas for all API endpoints.
Strict validation — invalid inputs never reach the model.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal, Any
from datetime import datetime


class CustomerInput(BaseModel):
    customerID: Optional[str] = None
    tenure: int = Field(..., ge=0, description="Months as customer (≥ 0)")
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)
    SeniorCitizen: Literal[0, 1]
    gender: Literal["Male", "Female"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]

    @field_validator("TotalCharges")
    @classmethod
    def total_ge_monthly(cls, v, info):
        monthly = info.data.get("MonthlyCharges", 0)
        tenure = info.data.get("tenure", 0)
        if tenure > 0 and v < monthly:
            raise ValueError("TotalCharges cannot be less than MonthlyCharges when tenure > 0")
        return v


class ShapFactor(BaseModel):
    feature: str
    value: str
    impact: str
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionOutput(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    risk_tier: Literal["Low", "Medium", "High"]
    prediction: Literal[0, 1]
    top_factors: List[ShapFactor]
    model_version: str
    prediction_id: str
    timestamp: datetime


class BatchInput(BaseModel):
    customers: List[CustomerInput] = Field(..., max_length=5000)


class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]
    total: int
    high_risk_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    model_version: str
    total_predictions: int
    predictions_last_24h: int
    drift_status: Literal["healthy", "mild_drift", "significant_drift"]
    last_drift_check: Optional[datetime]


class DriftStatusResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    status: Literal["healthy", "mild_drift", "significant_drift"]
    last_checked: Optional[datetime]
    report_available: bool
