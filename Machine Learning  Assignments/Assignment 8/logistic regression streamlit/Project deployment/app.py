# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 21:16:17 2025

@author: DELL
"""


import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and columns
@st.cache_resource
def load_model():
    """Loads the model and feature columns."""
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('model_columns.pkl', 'rb') as columns_file:
            model_columns = pickle.load(columns_file)
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Please run 'train_and_save_model.py' first.")
        st.stop()

# Get the model and columns
model, model_columns = load_model()

# --- App UI ---
st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# User inputs
pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0.42, 80.0, 30.0)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 512.33, 32.2)
embarked = st.radio("Port of Embarkation", ["C", "Q", "S"])

# Prediction button
if st.button("Predict"):
    # Create input DataFrame
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Fill in user data
    input_data['Age'] = age
    input_data['SibSp'] = sibsp
    input_data['Parch'] = parch
    input_data['Fare'] = fare
    
    # Handle categorical variables
    if sex == 'male':
        input_data['Sex_male'] = 1
    
    if pclass == 2:
        input_data['Pclass_2'] = 1
    elif pclass == 3:
        input_data['Pclass_3'] = 1

    if embarked == 'Q':
        input_data['Embarked_Q'] = 1
    elif embarked == 'S':
        input_data['Embarked_S'] = 1
        
    # Make prediction
    survival_prob = model.predict_proba(input_data)[0][1]
    
    # Display result
    st.write("---")
    if survival_prob > 0.5:
        st.success(f"Prediction: You would have survived! ({survival_prob:.2%})")
    else:
        st.error(f"Prediction: You might not have survived. ({survival_prob:.2%})")

