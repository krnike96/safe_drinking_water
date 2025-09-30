import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration for Custom Styles ---
st.set_page_config(
    page_title="Safe Water Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern, clean dark look
st.markdown("""
<style>
    /* 1. Global Streamlit Theme Override for Dark Mode */
    /* This forces the Streamlit structure to dark colors */
    :root {
        --base-background-color: #1c1c1c; /* Dark charcoal */
        --font-color: #f0f0f0; /* Light gray text */
    }
    
    /* 2. Main container styling */
    .stApp {
        background-color: #1c1c1c; /* Dark charcoal background */
        color: #f0f0f0; /* Light text */
    }

    /* Header styling */
    h1 {
        color: #00bcd4; /* Cyan/Teal heading for contrast */
        text-align: center;
        margin-bottom: 0.5em;
    }

    /* Subheader/Description styling */
    .stMarkdown p {
        color: #b0b0b0; /* Lighter gray description text */
        text-align: center;
        font-size: 1.1em;
    }

    /* 3. Button Styling for Dark Theme Contrast */
    /* Targeting the primary button class (which the prediction button uses) */
    .stButton>button {
        background-color: #00bcd4; /* Cyan button background */
        color: #1c1c1c; /* Dark text on button for max contrast */
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #008ba3; /* Darker cyan on hover */
        color: #f0f0f0;
    }

    /* Success/Error message styling */
    .stSuccess > div {
        background-color: #155724; /* Darker green background */
        color: #d4edda;
        border-color: #c3e6cb;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
    }
    .stError > div {
        background-color: #721c24; /* Darker red background */
        color: #f8d7da;
        border-color: #f5c6cb;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading (Crucial Step: Ensure paths are correct) ---
try:
    # Load the saved model and scaler from the 'Model' directory
    model = joblib.load('Model/best_rf_model.pkl')
    scaler = joblib.load('Model/scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'best_rf_model.pkl' and 'scaler.pkl' are in a folder named 'Model' in your project root.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading model artifacts: {e}")
    st.stop()


# --- UI Layout ---

st.title('Safe Drinking Water Predictor')
st.write('Input the water quality metrics below to instantly predict if the water is safe for drinking.')

# Instructions based on typical dataset ranges
# PH: (0 to 14)
# Hardness: (50 to 350 mg/L)
# Solids: (200 to 61000 mg/L)
# Chloramines: (0 to 14 mg/L)
# Sulfate: (100 to 500 mg/L)
# Conductivity: (200 to 800 μS/cm)
# Organic Carbon: (2 to 28 mg/L)
# Trihalomethanes: (0 to 125 μg/L)
# Turbidity: (0 to 6.5 NTU)


# Use columns for a two-column input layout
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input(
        'pH (Acidity/Alkalinity)', 
        min_value=0.0, 
        max_value=14.0, 
        value=7.0,
        help="Typical range is 6.5 to 8.5. Values outside 0-14 may destabilize the model."
    )
    
    solids = st.number_input(
        'Solids (Total Dissolved Solids)', 
        value=20000.0,
        help="Range: 200 mg/L to 61,000 mg/L. High solids indicate more impurities."
    )

    sulfate = st.number_input(
        'Sulfate (mg/L)', 
        value=330.0,
        help="Range: 100 mg/L to 500 mg/L. Recommended safe limit is often below 250 mg/L."
    )
    
    organic_carbon = st.number_input(
        'Organic Carbon (mg/L)', 
        value=14.0,
        help="Range: 2 mg/L to 28 mg/L. High levels can indicate contamination."
    )
    
    turbidity = st.number_input(
        'Turbidity (NTU)', 
        value=4.0,
        help="Range: 0 NTU to 6.5 NTU. Measures the cloudiness or haziness of water."
    )

with col2:
    hardness = st.number_input(
        'Hardness (mg/L)', 
        value=180.0,
        help="Range: 50 mg/L to 350 mg/L. High values indicate more dissolved minerals (Calcium/Magnesium)."
    )

    chloramines = st.number_input(
        'Chloramines (mg/L)', 
        value=7.0,
        help="Range: 0 mg/L to 14 mg/L. Used as a disinfectant; high levels can be harmful."
    )

    conductivity = st.number_input(
        'Conductivity (μS/cm)', 
        value=400.0,
        help="Range: 200 μS/cm to 800 μS/cm. Measures the ability of water to conduct electricity."
    )
    
    trihalomethanes = st.number_input(
        'Trihalomethanes (μg/L)', 
        value=66.0,
        help="Range: 0 μg/L to 125 μg/L. Byproducts of water disinfection."
    )
    # Placeholder to balance the layout
    st.write("") 
    st.write("") 


# Prediction Button (Centered)
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button('Predict Potability', key='predict_button'):
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
                              columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(scaled_input)

    # Display the result
    st.markdown("---")
    if prediction[0] == 1:
        st.success('✅ Prediction: Based on the provided metrics, the water is estimated to be SAFE TO DRINK.')
    else:
        st.error('⚠️ Prediction: Based on the provided metrics, the water is estimated to be NOT SAFE TO DRINK.')

st.markdown("</div>", unsafe_allow_html=True)
