import streamlit as st
import pandas as pd
import pickle

st.title("Telecom Customer Analysis")

# Load models
with open('logreg_model.pkl', 'rb') as f:
    logreg_model = pickle.load(f)

with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Select mode
mode = st.selectbox("Select Analysis Mode", 
                    ["Logistic Regression - Churn Prediction", 
                     "KNN - Churn Prediction",
                     "K-Means - Customer Segmentation"])

# Minimal inputs
gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
tenure_months = st.number_input("Tenure Months", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0)
fiber_optics_service = st.selectbox("Fiber Optics Service", [0, 1])
month_to_month_contract = st.selectbox("Month-to-month Contract", [0, 1])
credit_card_auto_paymentmethod = st.selectbox("Credit Card Auto Payment Method", [0, 1])

# Default features (0)
default_features = {
    'Partner': 0,
    'Dependents': 0,
    'Multiple Lines': 0,
    'Online Security': 0,
    'Online Backup': 0,
    'Device Protection': 0,
    'Tech Support': 0,
    'Streaming TV': 0,
    'Paperless Billing': 0,
    'DSL_Service': 0,
    'one-year_contract': 0,
    'two-year_contract': 0,
    'bank-transfer-auto_paymentmethod': 0,
    'electronic-check_paymentmethod': 0,
    'mailed-check_paymentmethod': 0
}

# Prepare full input dict with all features
input_dict = {
    'Gender': gender,
    'Senior Citizen': senior_citizen,
    'Tenure Months': tenure_months,
    'Monthly Charges': monthly_charges,
    'Total Charges': total_charges,
    'Fiber-Optics_Service': fiber_optics_service,
    'month-to-month_contract': month_to_month_contract,
    'credit-card-auto_paymentmethod': credit_card_auto_paymentmethod,
}

input_dict.update(default_features)

# Feature order for DataFrame
feature_order = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure Months', 'Multiple Lines',
    'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
    'Paperless Billing', 'Monthly Charges', 'Total Charges', 'Fiber-Optics_Service', 'DSL_Service',
    'month-to-month_contract', 'one-year_contract', 'two-year_contract',
    'bank-transfer-auto_paymentmethod', 'credit-card-auto_paymentmethod',
    'electronic-check_paymentmethod', 'mailed-check_paymentmethod'
]

input_df = pd.DataFrame([{k: input_dict[k] for k in feature_order}])

if st.button("Run Analysis"):
    if mode == "Logistic Regression - Churn Prediction":
        pred = logreg_model.predict(input_df)[0]
        st.write("Prediction:", "Will Churn" if pred == 1 else "Will Not Churn")

    elif mode == "KNN - Churn Prediction":
        pred = knn_model.predict(input_df)[0]
        st.write("Prediction:", "Will Churn" if pred == 1 else "Will Not Churn")

    else:  # K-Means Clustering
        cluster = kmeans_model.predict(input_df)[0]
        st.write(f"Customer belongs to cluster: {cluster}")
