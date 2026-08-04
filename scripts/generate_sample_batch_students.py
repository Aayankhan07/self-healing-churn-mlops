import pandas as pd
import random
from pathlib import Path

def generate_school_students_sample(n=100):
    grades = [f"Grade {i}" for i in range(1, 13)]
    engagement_levels = ["High", "Medium", "Low"]
    fee_statuses = ["Paid", "Pending", "Installment Plan"]
    
    rows = []
    for i in range(1, n + 1):
        grade = random.choice(grades)
        grade_num = int(grade.split()[1])
        attendance = round(random.uniform(60.0, 99.5), 1)
        gpa = round(random.uniform(1.8, 4.0), 2)
        engagement = random.choice(engagement_levels)
        fee = random.choice(fee_statuses)
        
        # Telecom compatible mappings for underlying model
        tenure_months = grade_num * 6
        monthly_fee = 150.0 + (grade_num * 20)
        total_fee = round(monthly_fee * tenure_months, 2)
        contract = "Month-to-month" if fee == "Pending" else ("One year" if grade_num <= 8 else "Two year")
        
        rows.append({
            "studentID": f"STU-{1000 + i}",
            "customerID": f"STU-{1000 + i}",
            "grade_level": grade,
            "attendance_percentage": attendance,
            "gpa_average": gpa,
            "parent_engagement": engagement,
            "tuition_status": fee,
            "tenure": tenure_months,
            "MonthlyCharges": monthly_fee,
            "TotalCharges": total_fee,
            "SeniorCitizen": 0,
            "gender": random.choice(["Male", "Female"]),
            "Partner": "Yes" if engagement == "High" else "No",
            "Dependents": "Yes",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL" if grade_num <= 5 else "Fiber optic",
            "OnlineSecurity": "Yes" if gpa > 3.0 else "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes" if engagement == "High" else "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)" if fee == "Paid" else "Electronic check",
        })
    
    df = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parents[1] / "data" / "sample_batch_students.csv"
    df.to_csv(out_path, index=False)
    print(f"School student sample batch CSV generated at: {out_path}")

if __name__ == "__main__":
    generate_school_students_sample()
