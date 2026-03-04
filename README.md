# Credit Risk Prediction System

## Overview
This project implements an end-to-end **credit risk prediction system** that estimates the probability of loan default and makes approval or rejection decisions using **cost-sensitive thresholding**.  
The emphasis is on business-aligned decision-making rather than raw accuracy.

The project covers data preprocessing, leakage prevention, model training, evaluation, and deployment using Flask inside a Docker container.

---

## Live Demo
The application is deployed and accessible here:

**creditriskpredictionsystem.up.railway.app**

The web app allows users to input loan application details and returns:
- Probability of default
- Final decision (APPROVE / REJECT) based on a cost-optimized threshold

---

![alt text](op1.png)
![alt text](op2.png)

---

## Project Flow

![alt text](flow.png)

---

## Project Structure

```
app/
├── app.py
├── credit_risk_model.pkl
├── feature_columns.pkl
└── templates/
    └── index.html
Dataset/
└── loans_full_schema.csv
notebook/
└── notebook.ipynb
├── Dockerfile
├── op1.png
├── op2.png
└── requirements.txt
```

---

## Problem Statement
Loan approval decisions involve asymmetric costs:
- **False Negative (FN):** Approving a borrower who later defaults (high cost)
- **False Positive (FP):** Rejecting a borrower who would have repaid (lower cost)

This project models default risk and applies **cost-based threshold optimization** instead of using accuracy or a fixed 0.5 cutoff.

---

## Data Processing
- Removed loans with status `Current`
- Consolidated loan status into `paid` and `default`
- Prevented data leakage by dropping post-loan outcome features
- Handled missing values using median imputation and indicator variables
- Engineered `credit_history_years` from credit start date
- Dropped redundant features (`grade` in favor of `sub_grade`)

---

## Modeling
Models evaluated:
- Logistic Regression  
- L1-Regularized Logistic Regression  
- Random Forest (baseline)

**Final Model:** L1-Regularized Logistic Regression  
Selected for better ROC-AUC, generalization, and stable probability estimates.

Class imbalance handled using stratified splitting and `class_weight="balanced"`.

---

## Evaluation Strategy
- **ROC-AUC** used for model selection
- Accuracy avoided due to class imbalance and asymmetric business costs

---

## Cost-Sensitive Thresholding
Instead of a default threshold, the decision threshold is optimized using:

**Total Cost = (False Negatives × Cost_FN) + (False Positives × Cost_FP)**

Assumptions:
- Cost_FN = 3  
- Cost_FP = 1  

The threshold minimizing total cost is used for final predictions.

---

## Docker Setup

The application is containerized using Docker for consistent and portable deployment.

**Dockerfile overview:**
- Base image: `python:3.11-slim`
- Installs `gcc` and `g++` for compiling any native Python dependencies
- Exposes port `5000`
- Installs dependencies from `requirements.txt`
- Runs as a non-root user for security
- Served via Gunicorn

---

## How to Run

### Option 1: Run with Docker (Recommended)

1. **Build the Docker image**
   ```bash
   docker build -t credit-risk .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 credit-risk
   ```

3. Open `http://localhost:5000` in a browser.

### Option 2: Run Locally (without Docker)

```bash
pip install -r requirements.txt
cd app
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

---

## Key Takeaways
- Credit risk modeling prioritizes decision quality over accuracy
- Probabilities must be converted into actions using business costs
- Preventing data leakage is essential in financial ML systems

---

## Limitations
- Cost assumptions are illustrative
- Default feature values are static
- No production monitoring or retraining pipeline implemented