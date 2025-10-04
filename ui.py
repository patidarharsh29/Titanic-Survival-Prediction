import streamlit as st
import joblib
import numpy as np

# Load trained Titanic model
model = joblib.load("model.pkl")

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details to check survival probability:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert categorical to numeric (same as training encoding)
sex = 1 if sex == "female" else 0
embarked_dict = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_dict[embarked]

# Create input array
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]  # no hasattr check
    
    if prediction[0] == 1:
        st.success("Prediction: Survived 🎉")
        st.info(f"Survival Probability: {proba:.2%}")
    else:
        st.error("Prediction: Did Not Survive 💀")
        st.info(f"Survival Probability: {proba:.2%}")