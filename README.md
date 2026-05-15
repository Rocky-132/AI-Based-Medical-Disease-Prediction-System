# AI-Based-Medical-Disease-Prediction-System
 AI-Based Medical Disease Prediction System using Machine Learning to predict Diabetes and Heart Disease risk from patient health parameters. Built with Python, Streamlit, Scikit-learn, and XGBoost with interactive UI, PDF report generation, model comparison, and real-time healthcare risk prediction.

 AI-Based Medical Disease Prediction System

 Overview

AI-Based Medical Disease Prediction System is a Machine Learning-powered healthcare application that predicts the risk of Diabetes and Heart Disease using patient medical parameters.

The project combines:

* Machine Learning classification models
* Interactive Streamlit web application
* PDF report generation
* Healthcare datasets
* Modern UI/UX design

This application helps users understand potential disease risks through predictive analytics and downloadable medical reports.

---

 Features

 Disease Prediction

 Diabetes Risk Prediction

Predicts diabetes probability using:

* Glucose level
* Blood pressure
* BMI
* Insulin
* Age
* Pregnancies
* Diabetes pedigree function

 Heart Disease Prediction

Predicts heart disease risk using:

* Chest pain type
* Cholesterol
* Blood pressure
* ECG results
* Heart rate
* Exercise-induced angina
* Age & gender

---

 Smart PDF Reports

* Downloadable prediction reports
* Includes patient inputs
* Risk probability
* AI-generated recommendations
* Timestamped medical summary

---

 Machine Learning Models Used

The project compares multiple ML algorithms:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* XGBoost

The best-performing model is automatically selected based on:

* Accuracy
* F1-Score
* Precision
* Recall

---

 Modern UI

* Streamlit-powered web interface
* Glassmorphism UI design
* Interactive forms
* Responsive layout
* Real-time predictions

---

 Project Structure

```bash id="med1"
medical-disease-prediction/
│
├── app.py
├── download_data.py
├── train_diabetes.py
├── train_heart.py
├── diabetes_model.pkl
├── heart_model.pkl
├── requirements.txt
├── README.md
└── datasets/
```

---

 Technologies Used

## Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

## Frontend/UI

* Streamlit

## Additional Tools

* Joblib
* FPDF2

---

 Datasets Used

## Diabetes Dataset

* Pima Indians Diabetes Dataset

## Heart Disease Dataset

* UCI Cleveland Heart Disease Dataset

---

 Installation

 Clone Repository

```bash id="med2"
git clone https://github.com/your-username/medical-disease-prediction.git
cd medical-disease-prediction
```

 Install Dependencies

```bash id="med3"
pip install -r requirements.txt
```

 Download Datasets

```bash id="med4"
python download_data.py
```

 Train Models

```bash id="med5"
python train_diabetes.py
python train_heart.py
```

 Run Streamlit App

```bash id="med6"
streamlit run app.py
```

---

Key Functionalities

✅ Disease risk prediction
✅ AI-based healthcare analytics
✅ Probability-based prediction
✅ PDF medical report generation
✅ ML model comparison
✅ Interactive medical forms
✅ Real-time prediction system

---

#  Future Enhancements

* Doctor appointment integration
* AI healthcare chatbot
* Cloud database storage
* Patient login system
* Deployment on AWS/Render
* Deep Learning models
* Medical image analysis

---

#  Disclaimer

This application is developed for educational and research purposes only. Predictions should not be considered professional medical advice. Always consult certified healthcare professionals for diagnosis and treatment.

---

# Author

Developed by Devendra Rakesh Vajrapu
