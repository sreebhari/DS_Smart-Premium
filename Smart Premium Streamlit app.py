import streamlit as st
import pandas as pd
import pickle
import joblib
from scipy import stats
from datetime import date
import numpy as np
import sklearn

# Load your saved model
model = joblib.load("best_model_XGBoost.pkl")
gender_obj=pickle.load(open("gender.pkl","rb"))
marital_status_obj=pickle.load(open('Marital_Status.pkl',"rb"))
education_obj=pickle.load(open('Education_Level.pkl',"rb"))
occupation_obj=pickle.load(open('Occupation.pkl',"rb"))
location_obj=pickle.load(open('Location.pkl',"rb"))
policy_type_obj=pickle.load(open('Policy_Type.pkl',"rb"))
feedback_obj=pickle.load(open('Customer_Feedback.pkl',"rb"))
smoking_status_obj=pickle.load(open('Smoking_Status.pkl',"rb"))
exercise_obj=pickle.load(open('Exercise_Frequency.pkl',"rb"))
property_type_obj=pickle.load(open('Property_Type.pkl',"rb"))
policy_start_obj=pickle.load(open('Policy_Start_Date_Clean.pkl',"rb"))
scale=pickle.load(open('scale.pkl',"rb"))

# Streamlit UI
st.set_page_config(page_title="Insurance Duration Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Insurance premium Prediction App")
st.markdown("### Enter Customer Details Below:")

# --- Input Form ---
with st.form("insurance_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        annual_inc = st.number_input("Annual Income", min_value=0, step=1000)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        dependents = st.number_input("Number of Dependents", min_value=0, step=1)
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
        duration = st.number_input("Insurance duration", min_value=1, max_value=10, step=1)

    with col2:
        occupation = st.text_input("Occupation", placeholder="e.g. Self-Employed, Engineer, Teacher")
        health_score = st.number_input("Health Score", min_value=0.0, max_value=100.0, step=0.1)
        location = st.selectbox("Location", ["Urban", "Rural", "Suburban"])
        policy_type = st.selectbox("Policy Type", ["Basic", "Premium", "Comprehensive"])
        prev_claims = st.number_input("Previous Claims", min_value=0, step=1)
        vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, step=1)

    with col3:
        credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, step=1)
        # policy_start_date = st.date_input("Policy Start Date", value=date.today())
        # policy_start_date = st.date_input("Policy Start Date", value=date.today()).strftime("%Y-%m-%d")
        policy_start_date = st.date_input("Policy Start Date", value=date.today())
        feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good", "Excellent"])
        smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
        exercise = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
        property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo", "Other"])

    submitted = st.form_submit_button("🔍 Predict Insurance Duration")

policy_start_date_str = policy_start_date.strftime("%Y-%m-%d")

def safe_encode_date(encoder, date_str):
    """Safely encode policy start date using LabelEncoder."""
    try:
        # Try normal transform
        return encoder.transform([date_str])[0]
    except ValueError:
        # Handle unseen dates gracefully
        # st.warning(f"⚠️ Date '{date_str}' was unseen during training. Added temporarily for encoding.")
        encoder.classes_ = np.append(encoder.classes_, date_str)
        return encoder.transform([date_str])[0]
    
gender_map=gender_obj.transform([gender])[0]
marital_status_map=marital_status_obj.transform([marital_status])[0]
education_map=education_obj.transform([education])[0]
occupation_map=occupation_obj.transform([occupation])[0]
location_map=location_obj.transform([location])[0]
policy_type_map=policy_type_obj.transform([policy_type])[0]
feedback_map=feedback_obj.transform([feedback])[0]
smoking_status_map=smoking_status_obj.transform([smoking_status])[0]
exercise_map=exercise_obj.transform([exercise])[0]
property_type__map=property_type_obj.transform([property_type])[0]
# policy_start_map = policy_start_obj.transform([policy_start.strftime("%Y-%m-%d")])[0]
# policy_start_map = policy_start_obj.transform([policy_start_date_str])[0]
policy_start_map = safe_encode_date(policy_start_obj, policy_start_date_str)

# --- Prediction ---
if submitted:
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender_map,
        'Annual Income': annual_inc,
        'Marital Status': marital_status_map,
        'Number of Dependents': dependents,
        'Education Level': education_map,
        'Occupation': occupation_map,
        'Health Score': health_score,
        'Location': location_map,
        'Policy Type': policy_type_map,
        'Previous Claims': prev_claims,
        'Vehicle Age': vehicle_age,
        'Credit Score': credit_score,
        'Insurance Duration': duration,
        'Customer Feedback': feedback_map,
        'Smoking Status': smoking_status_map,
        'Exercise Frequency': exercise_map,
        'Property Type': policy_type_map,
        'Policy_Start_Date_Clean': policy_start_map,
    }])

    input_df["Annual Income"] = stats.boxcox(input_df["Annual Income"],lmbda = 0.5)

    input_scaled=scale.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    st.success(f"🧮 Predicted Insurance Amount: **{prediction[0]:.2f} **")
