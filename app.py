import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
import base64
import os
import datetime

# --- Page Config & Styling ---
st.set_page_config(page_title="AI Medical Predictor", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

def load_models():
    diabetes_model = joblib.load('diabetes_model.pkl')
    heart_model = joblib.load('heart_model.pkl')
    return diabetes_model, heart_model

try:
    diabetes_model, heart_model = load_models()
except Exception as e:
    st.error("Error loading models. Please ensure models are trained and saved.")
    st.stop()

# Enhance UI with Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        background: linear-gradient(135deg, #1f1c2c 0%, #928DAB 100%);
    }
    .glass-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 30px;
        color: white;
        margin-bottom: 20px;
    }
    h1, h2, h3, p, label {
        color: #f0f2f6 !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .disclaimer {
        font-size: 12px;
        color: #ffcccc !important;
        text-align: center;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Define PDF Generator ---
def generate_pdf(prediction, probability, disease_type, inputs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AI-Based Medical Disease Prediction Report", ln=1, align='C')
    
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Inputs:", ln=1, align='L')
    pdf.set_font("Arial", '', 11)
    for key, value in inputs.items():
        pdf.cell(200, 8, txt=f"{key}: {value}", ln=1, align='L')
        
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Prediction Result:", ln=1, align='L')
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Disease: {disease_type}", ln=1, align='L')
    pdf.cell(200, 10, txt=f"Risk Level: {prediction}", ln=1, align='L')
    pdf.cell(200, 10, txt=f"Probability of being High Risk: {probability*100:.2f}%", ln=1, align='L')
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="DISCLAIMER: This prediction is only for educational purposes and should not be considered medical advice. Please consult a certified doctor for diagnosis.")
    
    # fpdf2 outputs a byte string when output() is called without a dest, or an object we can convert to bytes
    try:
        return bytes(pdf.output())
    except:
        return pdf.output(dest='S').encode('latin1')

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download Prediction Report (PDF)</a>'


# --- Main App ---
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Disease for Prediction:",
        ["Home", "Diabetes Prediction", "Heart Disease Prediction", "About & Models"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("🤖 AI-Based Medical Predictor via Machine Learning")
    
    if app_mode == "Home":
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.title("AI-Based Medical Disease Prediction System 🩺")
        st.write("Welcome to the Medical Disease Prediction System. Please use the sidebar to select the disease you want to assess.")
        st.write("This application was built using Machine Learning algorithms trained on public healthcare datasets to predict the likelihood of developing diabetes or heart disease.")
        
        st.subheader("Capabilities:")
        st.write("✅ **Diabetes Risk Prediction**: Utilizes the Pima Indians Diabetes Database.")
        st.write("✅ **Heart Disease Risk Prediction**: Utilizes the UCI Heart Disease dataset.")
        st.write("✅ **Insightful Reports**: Downloadable detailed PDF reports.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif app_mode == "Diabetes Prediction":
        st.title("🩸 Diabetes Risk Prediction")
        st.markdown("Enter your latest health parameters to check your diabetes risk probability.")
        with st.form("diabetes_form"):
            col1, col2 = st.columns(2)
            with col1:
                pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)
                glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=120)
                blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70)
                skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
            with col2:
                insulin = st.number_input('Insulin Level (IU/mL)', min_value=0, max_value=900, value=79)
                bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
                dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
                age = st.number_input('Age', min_value=0, max_value=120, value=30, step=1)
            
            submit = st.form_submit_button("Predict Diabetes Risk")
            
        if submit:
            features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            prediction = diabetes_model.predict(features)
            probability = diabetes_model.predict_proba(features)[0][1]
            
            st.markdown("### Prediction Results")
            if prediction[0] == 1:
                st.error(f"🔴 **High Risk of Diabetes** (Probability: {probability*100:.2f}%)")
                st.warning("Suggestion: Please consult a doctor immediately and monitor your blood sugar levels.")
            else:
                st.success(f"🟢 **Low Risk of Diabetes** (Probability: {probability*100:.2f}%)")
                st.info("Suggestion: Maintain a healthy lifestyle and diet!")
                
            inputs = {
                'Pregnancies': pregnancies, 'Glucose': glucose, 'Blood Pressure': blood_pressure,
                'Skin Thickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
                'Diabetes Pedigree Function': dpf, 'Age': age
            }
            pdf_bytes = generate_pdf("High Risk" if prediction[0] == 1 else "Low Risk", probability, "Diabetes", inputs)
            html = create_download_link(pdf_bytes, "Diabetes_Report.pdf")
            st.markdown(html, unsafe_allow_html=True)
            
    elif app_mode == "Heart Disease Prediction":
        st.title("❤️ Heart Disease Risk Prediction")
        st.markdown("Enter your health data to assess your heart disease risk.")
        with st.form("heart_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input('Age', min_value=1, max_value=120, value=45, step=1)
                sex = st.selectbox('Sex', [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
                cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
                trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, value=120)
                chol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
            with col2:
                fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                restecg = st.selectbox('Resting ECG Results', [0, 1, 2])
                thalach = st.number_input('Max Heart Rate Achieved', min_value=50, max_value=250, value=150)
                exang = st.selectbox('Exercise Induced Angina', [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                oldpeak = st.number_input('ST Depression Induced', min_value=0.0, max_value=10.0, value=1.0)
            with col3:
                slope = st.selectbox('Slope of the peak exercise ST segment', [0, 1, 2])
                ca = st.number_input('Number of major vessels (0-3)', min_value=0, max_value=4, value=0, step=1)
                thal = st.selectbox('Thalassemia', [0, 1, 2, 3], help="1=normal, 2=fixed defect, 3=reversable defect")
            
            submit = st.form_submit_button("Predict Heart Risk")
            
        if submit:
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction = heart_model.predict(features)
            probability = heart_model.predict_proba(features)[0][1]
            
            st.markdown("### Prediction Results")
            if prediction[0] == 1:
                st.error(f"🔴 **High Risk of Heart Disease** (Probability: {probability*100:.2f}%)")
                st.warning("Suggestion: Please seek immediate medical consultation and avoid intense physical exertion.")
            else:
                st.success(f"🟢 **Low Risk of Heart Disease** (Probability: {probability*100:.2f}%)")
                st.info("Suggestion: Keep up the good work! Excerise regularly and eat healthy.")
                
            inputs = {
                'Age': age, 'Sex': "Male" if sex==1 else "Female", 'Chest Pain': cp,
                'Blood Pressure': trestbps, 'Cholesterol': chol, 'Fasting Blood Sugar > 120': "Yes" if fbs==1 else "No",
                'ECG Result': restecg, 'Max Heart Rate': thalach, 'Exercise Angina': "Yes" if exang==1 else "No",
                'ST Depression': oldpeak, 'Slope': slope, 'Major Vessels': ca, 'Thalassemia': thal
            }
            pdf_bytes = generate_pdf("High Risk" if prediction[0] == 1 else "Low Risk", probability, "Heart Disease", inputs)
            html = create_download_link(pdf_bytes, "Heart_Disease_Report.pdf")
            st.markdown(html, unsafe_allow_html=True)
            
    elif app_mode == "About & Models":
        st.title("📊 Model Comparison & Details")
        st.write("This section details the machine learning models trained for these predictions.")
        st.write("**Diabetes Dataset:** Pima Indians Diabetes Database (768 records, 8 numeric attributes).")
        st.write("**Heart Disease Dataset:** UCI Cleveland Heart Disease Dataset (303 records, 13 attributes).")
        
        st.subheader("Model Selection Process")
        st.write("We trained standard algorithms: Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost. The models saved and used in this application are the ones with the highest Accuracy and F1-score on a 20% test split.")
        
    st.markdown('<p class="disclaimer">🚨 DISCLAIMER: This prediction is only for educational purposes and should not be considered medical advice. Please consult a certified doctor for diagnosis.</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
