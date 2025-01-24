import streamlit as st
import pickle
import pandas as pd

# Load the saved pipeline
@st.cache_resource
def load_pipeline():
    with open('lgb_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

# Load the pipeline
pipeline = load_pipeline()

# Streamlit app UI
st.title("LightGBM Prediction App")
st.write("Enter input features to get predictions using the LightGBM model.")

# Define input fields for features
feature1 = st.selectbox("Feature1 (Categorical)", options=['A', 'B', 'C'])
feature2 = st.number_input("Feature2 (Numerical)", value=1.0)

# Create input DataFrame
input_data = pd.DataFrame({'feature1': [feature1], 'feature2': [feature2]})

# Prediction button
if st.button("Predict"):
    # Make predictions
    prediction = pipeline.predict(input_data)
    predicted_class = prediction[0]
    
    # Display prediction
    st.write(f"Predicted Class: {predicted_class}")
