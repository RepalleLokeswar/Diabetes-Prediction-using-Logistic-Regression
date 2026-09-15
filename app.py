import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS FOR STYLING
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Header section styling */
    .header-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    
    /* Custom button styling */
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 0.6rem;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #004999;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('logistic_regression_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("Error: Could not load `logistic_regression_model.pkl`. Please ensure it is in the same directory.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=100)
    st.title("About the App")
    st.info(
        "This application uses a **Logistic Regression** machine learning model "
        "trained on clinical health parameters to estimate the probability of diabetes."
    )
    st.markdown("---")
    st.caption("Developed for Data Science Assignment 8")

# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="header-box">
        <h1 style="margin:0; color:#1f2937;">🩺 Diabetes Risk Assessment</h1>
        <p style="margin:5px 0 0 0; color:#4b5563;">Enter clinical measurements below to evaluate diabetes likelihood.</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INPUT FORM (2 COLUMNS)
# -----------------------------------------------------------------------------
st.subheader("📋 Patient Parameters")

col1, col2 = st.columns(2, gap="large")

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, help="Number of times pregnant")
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=200.0, value=120.0, help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)")
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=140.0, value=70.0, help="Diastolic blood pressure")
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, help="Triceps skin fold thickness")

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=900.0, value=79.0, help="2-Hour serum insulin")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=30.0, help="Body mass index (weight in kg/(height in m)^2)")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, format="%.3f", help="Scores likelihood of diabetes based on family history")
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)

st.markdown("---")

# -----------------------------------------------------------------------------
# PREDICTION & RESULTS SECTION
# -----------------------------------------------------------------------------
if st.button("Analyze Risk"):
    # Prepare input feature array
    input_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Make predictions
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    st.subheader("📊 Assessment Results")
    
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(
            label="Calculated Risk Probability",
            value=f"{probability:.1%}"
        )

    with res_col2:
        if prediction == 1:
            st.error("### ⚠️ High Risk Detected")
            st.write("The model predicts a higher likelihood of diabetes based on the provided clinical parameters. Further evaluation by a medical professional is recommended.")
        else:
            st.success("### ✅ Low Risk Detected")
            st.write("The model predicts a low likelihood of diabetes based on the provided parameters. Maintain a healthy lifestyle and regular checkups.")