import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Download and load the trained model pipeline
model_path = hf_hub_download(
    repo_id="vinodtigadi/tourism-package",
    repo_type="model",
    filename="tourism_project_model_v1.joblib"
)


model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction App")
st.write("""
This app predicts whether a customer will purchase a 'Tourism Package' product based on their profile and interaction data.
""")

# --- User Inputs ---

# Binary Inputs
gender = st.radio("Gender", ["Male", "Female"])
own_car = st.radio("Owns Car?", [0, 1], index=0)
passport = st.radio("Holds Passport?", [0, 1], index=1)

# Categorical Inputs
typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Freelancer", "Salaried", "Small Business", "Large Business"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"] )
designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "Senior Manager", "VP"])

# Numeric Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
city_tier = st.number_input("City Tier", min_value=1, max_value=3, value=2)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=50000)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=200, value=10)
number_of_person_visiting = st.number_input("Number of People Visiting", min_value=0, max_value=8, value=1)
preferred_property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=2)
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=10, value=5)
number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)

# Convert Gender to binary
gender_encoded = 0 if gender == "Male" else 1

# Collect inputs into DataFrame
input_data = pd.DataFrame([{
    "Gender": gender_encoded,
    "Passport": passport,
    "OwnCar": own_car,
    "TypeofContact": typeofcontact,
    "Occupation": occupation,
    "MaritalStatus": marital_status,
    "ProductPitched": product_pitched,
    "Designation": designation,
    "Age": age,
    "CityTier": city_tier,
    "MonthlyIncome": monthly_income,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "NumberOfTrips": number_of_trips,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "NumberOfFollowups": number_of_followups
}])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]  # probability of class 1

    result_text = "Customer will purchase ✅" if prediction == 1 else "Customer will NOT purchase ❌"

    st.subheader("Prediction")
    st.write(result_text)
    st.subheader("Probability of Purchase")
    st.write(f"{prediction_proba:.2%}")
