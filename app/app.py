from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and feature columns
model = joblib.load("credit_risk_model.pkl")

# Default values for internal features
DEFAULT_FEATURES = {
    "months_since_last_credit_inquiry": 12,
    "msci_missing": 0,
    "num_total_cc_accounts": 5,
    "num_open_cc_accounts": 3,
    "num_mort_accounts": 1,
    "account_never_delinq_percent": 90,
    "tax_liens": 0,
    "public_record_bankrupt": 0,
    "sub_grade": "C3",
    "state": "CA",
    "application_type": "Individual",
    "issue_month": "Jan-2020",
    "initial_listing_status": "w",
    "disbursement_method": "Cash",
    "inquiries_last_12m": 1,
    "delinq_2y": 0,
    "total_credit_lines": 10,
    "open_credit_lines": 5,
    "num_satisfactory_accounts": 10,
    "num_historical_failed_to_pay": 0,
    "accounts_opened_24m": 2,
    "num_cc_carrying_balance": 2,
    "num_active_debit_accounts": 1,
    "total_credit_limit": 20000,
    "total_debit_limit": 5000,
    "current_installment_accounts": 1
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    prediction = None
    probability = None

    if request.method == "POST":
        # Collect user inputs
        input_data = {
            "annual_income": float(request.form["annual_income"]),
            "debt_to_income": float(request.form["debt_to_income"]),
            "emp_length": float(request.form["emp_length"]),
            "loan_amount": float(request.form["loan_amount"]),
            "term": int(request.form["term"]),
            "interest_rate": float(request.form["interest_rate"]),
            "installment": float(request.form["installment"]),
            "homeownership": request.form["homeownership"],
            "verified_income": request.form["verified_income"],
            "loan_purpose": request.form["loan_purpose"],
            "credit_history_years": float(request.form["credit_history_years"]),
        }

        # Add internal defaults
        input_data.update(DEFAULT_FEATURES)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict probability
        prob_default = model.predict_proba(input_df)[0][1]

        # Decision
        prediction = "REJECT" if prob_default >= 0.45 else "APPROVE"
        probability = round(prob_default * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
    )

if __name__ == "__main__":
    app.run(debug=False)
