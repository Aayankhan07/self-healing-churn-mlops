"""
Pydantic v2 input/output schemas for all API endpoints.
Strict validation — invalid inputs never reach the model.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List, Literal
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
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    @field_validator("TotalCharges")
    @classmethod
    def total_ge_monthly(cls, v, info):
        monthly = info.data.get("MonthlyCharges", 0)
        tenure = info.data.get("tenure", 0)
        if tenure > 0 and v < monthly:
            raise ValueError(
                "TotalCharges cannot be less than MonthlyCharges when tenure > 0"
            )
        return v


class GenericCustomerInput(BaseModel):
    """
    Request model for domains that are not telecom.

    Their fields come from the domain's own baseline data rather than the
    telecom schema, so validation is permissive about extra keys and only
    enforces what the domain's spec actually declares. Healing has already run
    by the time a record reaches this model.
    """

    model_config = ConfigDict(extra="allow")

    customerID: Optional[str] = None

    def model_dump(self, **kwargs):
        # Keep the extras — they are the domain's real features.
        kwargs.setdefault("exclude_none", False)
        return super().model_dump(**kwargs)


def build_request_model(spec) -> type[BaseModel]:
    """
    Return the Pydantic model that validates requests for `spec`.

    Telecom keeps the hand-written CustomerInput so its OpenAPI schema and its
    strict Literal validation are unchanged for existing clients.
    """
    if spec.key == "telecom":
        return CustomerInput
    return GenericCustomerInput


def unknown_fields(record: dict, spec) -> List[str]:
    """
    Fields the caller sent that this domain's spec does not declare.

    They are accepted — a domain's baseline rarely covers every column its
    callers send — but reported as a data-quality event. Silently swallowing
    unrecognized fields is how the telecom-only schema went unnoticed for so
    long: a school request full of unknown keys still scored happily.
    """
    if not spec.field_names:
        # A permissive spec declares nothing, so nothing can be unexpected.
        return []
    known = set(spec.field_names) | {spec.id_column, spec.target_column, "domain"}
    return sorted(key for key in record if key not in known)


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
    healed_actions: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    time_to_churn_days: Optional[int] = None
    risk_horizon_summary: Optional[str] = None
    survival_timeline: Optional[dict] = None


class BatchInput(BaseModel):
    customers: List[CustomerInput] = Field(..., max_length=5000)


class LaxBatchInput(BaseModel):
    customers: List[dict] = Field(..., max_length=5000)


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
    # True when this domain serves artifacts copied from another domain rather
    # than a model trained on its own data.
    demo_fixture: bool = False


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
