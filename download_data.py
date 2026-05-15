import pandas as pd
import os

os.makedirs("datasets", exist_ok=True)

print("Downloading Diabetes Dataset...")
diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
diabetes_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df_diabetes = pd.read_csv(diabetes_url, header=None, names=diabetes_cols)
df_diabetes.to_csv("datasets/diabetes.csv", index=False)
print(f"Diabetes dataset saved. Shape: {df_diabetes.shape}")

print("Downloading Heart Disease Dataset...")
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df_heart = pd.read_csv(heart_url, header=None, names=heart_cols, na_values="?")
df_heart.dropna(inplace=True)
# Target has values 0-4, 0 is no disease, >0 is disease
df_heart['target'] = df_heart['target'].apply(lambda x: 1 if x > 0 else 0)
df_heart.to_csv("datasets/heart.csv", index=False)
print(f"Heart dataset saved. Shape: {df_heart.shape}")
