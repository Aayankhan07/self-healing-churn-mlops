"""
Automated Retention Action Playbooks Module.
Translates SHAP drivers and customer features into domain-specific intervention playbooks.
"""

from typing import List, Dict, Any


def generate_retention_playbook(
    domain_id: str,
    top_factors: List[Dict[str, Any]],
    features: Dict[str, Any],
    risk_tier: str,
) -> List[str]:
    """
    Generate domain-tailored retention action recommendations based on top SHAP drivers and raw features.
    """
    if risk_tier == "Low":
        return ["Account healthy. Maintain standard service delivery."]

    domain = str(domain_id).lower().strip()
    actions = []
    factor_names = [f.get("feature", "").lower() for f in (top_factors or [])]
    factor_str = " ".join(factor_names)

    if domain == "school":
        if (
            "attendance" in factor_str
            or "grade" in factor_str
            or "tenure" in factor_str
        ):
            actions.append(
                "Schedule mandatory Academic Counselor check-in and notify parents via portal."
            )
        if "fee" in factor_str or "payment" in factor_str or "contract" in factor_str:
            actions.append(
                "Offer deferred tuition installment plan or financial aid review."
            )
        if "support" in factor_str or "tech" in factor_str:
            actions.append(
                "Assign dedicated peer tutor and provide digital learning materials."
            )
        if not actions:
            actions.append(
                "Schedule proactive student progress review with homeroom teacher."
            )

    elif domain == "fitness":
        if "visit" in factor_str or "tenure" in factor_str or "contract" in factor_str:
            actions.append(
                "Send automated SMS offering a complimentary 1-on-1 Personal Trainer session."
            )
        if "monthlycharges" in factor_str or "payment" in factor_str:
            actions.append("Offer 15% discount on annual membership extension.")
        if not actions:
            actions.append(
                "Invite member to VIP group fitness class or wellness evaluation."
            )

    elif domain == "telecom":
        contract = str(features.get("Contract", "")).lower()
        monthly = float(features.get("MonthlyCharges", 0) or 0)

        if "contract" in factor_str or "month-to-month" in contract:
            actions.append(
                "Trigger proactive 10% loyalty discount on 1-year contract pitch."
            )
        if "monthlycharges" in factor_str or monthly > 70.0:
            actions.append("Offer bundle discount on high-speed fiber internet.")
        if "techsupport" in factor_str or "security" in factor_str:
            actions.append(
                "Offer free 3-month trial of Tech Support & Online Security Suite."
            )
        if not actions:
            actions.append("Schedule proactive customer retention outreach call.")

    elif domain == "ecommerce":
        if (
            "inactivity" in factor_str
            or "orders" in factor_str
            or "tenure" in factor_str
        ):
            actions.append(
                "Dispatch 15% win-back coupon code with free express shipping."
            )
        if "price" in factor_str or "monthlycharges" in factor_str:
            actions.append("Enroll customer in VIP Loyalty Cashback Program.")
        if not actions:
            actions.append(
                "Send personalized product recommendations based on browsing history."
            )

    else:
        actions.append(
            "Initiate proactive customer success outreach call within 24 hours."
        )
        actions.append("Offer tailored renewal incentive or service tier optimization.")

    return actions
