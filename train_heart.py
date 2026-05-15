import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

def load_data():
    df = pd.read_csv('datasets/heart.csv')
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    print("--- Heart Disease Model Training ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name} -> Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with Accuracy {best_accuracy:.4f}")
    joblib.dump(best_model, 'heart_model.pkl')
    print("Saved as heart_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
