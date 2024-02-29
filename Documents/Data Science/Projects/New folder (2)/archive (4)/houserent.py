import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
#st.set_page_config(page_title='Prediction', layout='wide')
loaded_model = tf.keras.models.load_model("C:/Users/DELL/Documents/Data Science/Projects/New folder (2)/archive (4)/saved_model.pb")

# loaded_model = tf.keras.models.load_model(r"C:\Users\DELL\Documents\Data Science\Projects\New folder (2)\archive (4)\saved_model.pb")
# Function to make predictions
def predict_rent(num_bhk, house_size, area_type, city_pin_code, furnishing_status, tenant_type, num_bathrooms):
    input_features = np.array([[num_bhk, house_size, area_type, city_pin_code, furnishing_status, tenant_type, num_bathrooms]])
    predicted_rent = loaded_model.predict(input_features)
    return predicted_rent

# Streamlit App
st.title("Housing Rent Predictor")

# User input section
st.header("Enter House Details")
num_bhk = st.number_input("Number of Bedrooms, Hall, Kitchen (BHK)", min_value=1, step=1)
house_size = st.number_input("Size of the House (in Square Feet)", min_value=1, step=1)
area_type = st.selectbox("Area Type", [1, 2, 3], format_func=lambda x: "Super Area" if x == 1 else ("Carpet Area" if x == 2 else "Built Area"))
city_pin_code = st.number_input("Pin Code of the City", min_value=1000, step=1)
furnishing_status = st.selectbox("Furnishing Status", [0, 1, 2], format_func=lambda x: ["Unfurnished", "Semi-Furnished", "Furnished"][x])
tenant_type = st.selectbox("Tenant Type", [1, 2, 3], format_func=lambda x: ["Bachelors", "Bachelors/Family", "Only Family"][x])
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)

# Prediction and display result
if st.button("Predict Rent"):
    predicted_rent = predict_rent(num_bhk, house_size, area_type, city_pin_code, furnishing_status, tenant_type, num_bathrooms)
    st.success(f"Predicted House Rent: {predicted_rent[0]:.2f}")
