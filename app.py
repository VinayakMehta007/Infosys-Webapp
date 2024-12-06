import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.impute import SimpleImputer



# Load the pre-trained model, scaler, label encoders, and feature names
def load_model():
    model = joblib.load("logistic_model.pkl")  # Load the Logistic Regression model
    scaler = joblib.load("scaler.pkl")  # Load the scaler
    label_encoders = joblib.load("label_encoders.pkl")  # Load the label encoders
    feature_names = joblib.load("feature_names.pkl")  # Load the saved feature names
    return model, scaler, label_encoders, feature_names

# Prediction function
def predict_stroke(inputs, model, scaler, label_encoders, feature_names):
    # Convert input to a DataFrame
    input_df = pd.DataFrame([inputs])

    # Standardize column names to match training data (convert to lowercase)
    input_df.columns = input_df.columns.str.lower()

    # Ensure the input dataframe has the same columns as the model's features
    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Add missing columns with default values (e.g., 0)

    # Reorder columns to match the feature names
    input_df = input_df[feature_names]

    # Handle missing values in the input (same as we did during training)
    numerical_columns = input_df.select_dtypes(include=['float64', 'int64']).columns
    cat_imputer = SimpleImputer(strategy="most_frequent")
    input_df[numerical_columns] = cat_imputer.fit_transform(input_df[numerical_columns])

    # Apply label encoding to categorical features
    for col in input_df.select_dtypes(include=['object']).columns:
        le = label_encoders.get(col)
        if le:
            input_df[col] = le.transform(input_df[col])

    # Scale the features
    input_scaled = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1]  # Probability for class 1

    return prediction, probability

# Streamlit UI
def main():
    # Set page title and background color
    st.set_page_config(page_title="Stroke Prediction App", page_icon=":guardsman:", layout="wide")

    # Apply custom CSS for styling
    st.markdown("""
        <style>
        .title {
            font-size: 40px;
            color: #ff6347; 
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        .subheader {
            font-size: 24px;
            color: #32cd32;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        .text {
            font-size: 18px;
            color: #1e90ff;
            font-family: 'Arial', sans-serif;
        }
        .prediction-result {
            font-size: 22px;
            color: #ff1493;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .header {
            background-color: #f0f8ff;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Page header
    st.markdown('<div class="header"><h1 class="title">Stroke Prediction App</h1></div>', unsafe_allow_html=True)
    
    # Inputs for prediction
    inputs = {
        'gender': st.selectbox('Gender', ['Male', 'Female']),
        'age': st.number_input('Age', min_value=0, max_value=100, value=30),
        'hypertension': st.selectbox('Hypertension', ['Yes', 'No']),
        'heart_disease': st.selectbox('Heart Disease', ['Yes', 'No']),
        'ever_married': st.selectbox('Ever Married', ['Yes', 'No']),
        'work_type': st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']),
        'residence_type': st.selectbox('Residence Type', ['Urban', 'Rural']),
        'avg_glucose_level': st.number_input('Average Glucose Level', value=85.0),
        'bmi': st.number_input('BMI', value=25.0),
        'smoking_status': st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown']),
    }

    # Load the model, scaler, label encoders, and feature names
    model, scaler, label_encoders, feature_names = load_model()

    # Process inputs for prediction
    processed_inputs = {
        'gender': inputs['gender'],
        'age': inputs['age'],
        'hypertension': 1 if inputs['hypertension'] == 'Yes' else 0,
        'heart_disease': 1 if inputs['heart_disease'] == 'Yes' else 0,
        'ever_married': 1 if inputs['ever_married'] == 'Yes' else 0,
        'work_type': inputs['work_type'],
        'residence_type': inputs['residence_type'],
        'avg_glucose_level': inputs['avg_glucose_level'],
        'bmi': inputs['bmi'],
        'smoking_status': inputs['smoking_status']
    }

    # Make prediction when button is pressed
    if st.button('Predict'):
        # Make prediction
        prediction, probability = predict_stroke(processed_inputs, model, scaler, label_encoders, feature_names)

        # Display output in a colorful and stylish format
        if prediction[0] == 1:
            st.markdown(f'<div class="prediction-result">The person is at risk of having a stroke!</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="text">Probability: <b>{probability[0]:.2f}</b></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-result">The person is not at risk of having a stroke.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="text">Probability: <b>{probability[0]:.2f}</b></div>', unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
