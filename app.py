import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model and scaler
model = joblib.load('Model/best_rf_model.pkl')
scaler = joblib.load('Model/scaler.pkl')

# Set up the title and description of the app
st.title('Safe Drinking Water Prediction')
st.write('Enter the water quality parameters to predict if the water is safe for consumption.')

# Create input fields for each feature
ph = st.number_input('pH', min_value=0.0, max_value=14.0)
hardness = st.number_input('Hardness')
solids = st.number_input('Solids')
chloramines = st.number_input('Chloramines')
sulfate = st.number_input('Sulfate')
conductivity = st.number_input('Conductivity')
organic_carbon = st.number_input('Organic Carbon')
trihalomethanes = st.number_input('Trihalomethanes')
turbidity = st.number_input('Turbidity')

# Create a button for prediction
if st.button('Predict'):
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
                              columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(scaled_input)

    # Display the result
    if prediction[0] == 1:
        st.success('Prediction: The water is safe to drink.')
    else:
        st.error('Prediction: The water is not safe to drink.')