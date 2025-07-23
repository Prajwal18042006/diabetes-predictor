import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('diabetes.csv')  # You must have this file

# Prepare data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Save model
joblib.dump(model, 'diabetes_model.pkl')

# Streamlit UI
st.title("🧠 Diabetes Prediction App")
st.markdown("Enter the details below:")

# Input features
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
Age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    # Load model
    model = joblib.load('diabetes_model.pkl')
    
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                            BMI, DiabetesPedigreeFunction, Age]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🩸 The person is **DIABETIC**")
    else:
        st.success("✅ The person is **NOT diabetic**")
