import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -------------------------------------------------
# Load model from Hugging Face
# -------------------------------------------------
model_path = hf_hub_download(
    repo_id="kalrap/M10_AIML_MLOps_Tourism_Project",
    filename="best_tourism_failure_model_v1.joblib"
)
model = joblib.load(model_path)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("Wellness Tourism Package Prediction App")

st.write("""
This application helps identify customers who are **likely to purchase a wellness tourism package**.
It enables data-driven targeting to improve marketing effectiveness and business growth.
""")

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])

with col2:
    num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=2)
    passport = st.selectbox("Passport", [0, 1])
    pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Own Car", [0, 1])
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    monthly_income = st.number_input("Monthly Income", min_value=5000, max_value=200000, value=30000)

st.header("Demographic Information")

type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])

# -------------------------------------------------
# Assemble Input DataFrame
# -------------------------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "PreferredPropertyStar": preferred_star,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "TypeofContact": type_of_contact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital_status
}])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Purchase Probability"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"

    st.subheader("Prediction Result")
    st.success(f"**{result}**")
    st.info(f"Purchase Probability: **{probability:.2%}**")
