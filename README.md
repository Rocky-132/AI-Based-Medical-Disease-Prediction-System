# AI-Based Medical Disease Prediction System

AI-Based Medical Disease Prediction System using Machine Learning to predict diabetes and heart disease risk from patient health parameters. Built with Python, Scikit-learn, Streamlit, and trained classification models with accuracy metrics and interactive web deployment.

## Features
- **Diabetes Risk Prediction**: Utilizes the Pima Indians Diabetes Database.
- **Heart Disease Risk Prediction**: Utilizes the UCI Heart Disease dataset.
- **Interactive UI**: Built with Streamlit with a modern Glassmorphism design.
- **Detailed PDF Reports**: Downloadable prediction results for reference.
- **Model Evaluation**: Employs multiple algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost) and selects the best performer.

## Tech Stack
- **Machine Learning**: Python, Pandas, NumPy, Scikit-learn, XGBoost
- **Frontend/UI**: Streamlit
- **Model Storage**: Joblib (Pickle)
- **PDF Generation**: fpdf2

## Installation

1. Clone this repository (or copy the project folder).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the initial data download and model training (Optional, pre-trained `.pkl` files might already be included):
   ```bash
   python download_data.py
   python train_diabetes.py
   python train_heart.py
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Disclaimer
🚨 **This prediction is only for educational purposes and should not be considered medical advice. Please consult a certified doctor for diagnosis.**
