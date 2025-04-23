import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy

# Load the trained model (pipe.pkl)
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Function to predict risk (good/bad) using the model
def predict_risk(input_data):
    # Convert the input data into a DataFrame for model prediction
    input_df = pd.DataFrame([input_data])
    prediction = pipe.predict(input_df)  # Get the prediction from the pipeline
    return 'Good' if prediction[0] == 1 else 'Bad'

# Streamlit UI
st.title('Credit Risk Prediction')
st.write("Enter the details below to predict the credit risk:")

# Input fields for user to provide details
age = st.number_input('Age', min_value=18, max_value=100, value=30)
sex = st.selectbox('Sex', options=['male', 'female'])
job = st.selectbox('Job', options=[1, 2, 3, 4])
housing = st.selectbox('Housing', options=['own', 'free'])
saving_accounts = st.selectbox('Saving accounts', options=['little', 'moderate', 'rich', 'no known savings'])
checking_account = st.selectbox('Checking account', options=['little', 'moderate', 'rich', 'no known checking'])
credit_amount = st.number_input('Credit amount', min_value=100, max_value=100000, value=5000)
duration = st.number_input('Duration', min_value=1, max_value=60, value=12)
purpose = st.selectbox('Purpose', options=['business', 'radio/TV', 'furniture/equipment', 'education'])

# Button to trigger prediction
if st.button('Predict'):
    # Prepare input data
    input_data = {
        'Age': age,
        'Sex': sex,
        'Job': job,
        'Housing': housing,
        'Saving accounts': saving_accounts,
        'Checking account': checking_account,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': purpose
    }
    
    # Get the prediction
    result = predict_risk(input_data)
    
    # Display the result
    st.write(f"Prediction: {result}")
