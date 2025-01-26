import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Load the pre-trained model and label encoders
model = load("lightgbm_model_method2.joblib")
label_encoders = load('label_encoders_method2.joblib')

# Load the data (replace with the actual path to the Excel file)
data = pd.read_excel("./usecase_4_.xlsx")

study_design_split = data['Study Design'].str.split('|', expand=True)
data['Allocation'] = study_design_split[0]
data['Intervention Model'] = study_design_split[1]
data['Primary Purpose'] = study_design_split[2]
data['Masking'] = study_design_split[3]

data = data.drop(columns=['Study Design'])

# Streamlit app interface
st.title('Recruitment Rate Prediction App')

# Form for user inputs
with st.form(key='prediction_form'):
    # Input fields for user data using selectbox from dataframe values
    study_status_options = data['Study Status'].dropna().unique()
    study_results_options = data['Study Results'].dropna().unique()
    sex_options = data['Sex'].dropna().unique()
    funder_type_options = data['Funder Type'].dropna().unique()
    study_type_options = data['Study Type'].dropna().unique()
    allocation_options = data['Allocation'].dropna().unique()
    intervention_model_options = data['Intervention Model'].dropna().unique()
    primary_purpose_options = data['Primary Purpose'].dropna().unique()
    masking_options = data['Masking'].dropna().unique()
    conditions_data = data['Conditions'].str.split('|').explode().unique()
    interventions_data = data['Interventions'].str.split('|').explode().unique()

    # User input fields
    study_status = st.selectbox('Study Status', study_status_options)
    study_results = st.selectbox('Study Results', study_results_options)
    conditions = st.multiselect('Conditions', conditions_data)
    interventions = st.multiselect('Interventions', interventions_data)
    sex = st.selectbox('Sex', sex_options)
    age = st.number_input('Age', min_value=18, max_value=100, step=1)
    phases = st.selectbox('Phases', data['Phases'].dropna().unique())
    enrollment = st.number_input('Enrollment', min_value=1, max_value=10000, step=1)
    funder_type = st.selectbox('Funder Type', funder_type_options)
    study_type = st.selectbox('Study Type', study_type_options)
    locations = st.text_input('Locations')
    study_duration = st.number_input('Study Duration (days)', min_value=1, max_value=10000, step=1)
    allocation = st.selectbox('Allocation', allocation_options)
    intervention_model = st.selectbox('Intervention Model', intervention_model_options)
    primary_purpose = st.selectbox('Primary Purpose', primary_purpose_options)
    masking = st.selectbox('Masking', masking_options)

    # Submit button
    submit_button = st.form_submit_button(label='Predict Recruitment Rate')

# Prepare the user input data
if submit_button:
    user_data = pd.DataFrame({
        'Study Status': [study_status],
        'Study Results': [study_results],
        'Conditions': ['|'.join(conditions)],
        'Interventions': ['|'.join(interventions)],
        'Sex': [sex],
        'Age': [age],
        'Phases': [phases],
        'Enrollment': [enrollment],
        'Funder Type': [funder_type],
        'Study Type': [study_type],
        'Locations': [locations],
        'Study_Duration': [study_duration],
        'Allocation': [allocation],
        'Intervention Model': [intervention_model],
        'Primary Purpose': [primary_purpose],
        'Masking': [masking]
    })

    # Encode the user input data
    def encode_categorical_data(data, label_encoders):
        cleaned_data = data.copy()
        cleaned_data_categorical_columns = cleaned_data.select_dtypes(include=['object']).columns

        for col in cleaned_data_categorical_columns:
            if col in label_encoders:
                le = label_encoders[col]
                le_classes = set(le.classes_)
                new_classes = set(cleaned_data[col].unique()) - le_classes

                if new_classes:
                    le.classes_ = np.append(le.classes_, list(new_classes))

                cleaned_data[col] = le.transform(cleaned_data[col])
            else:
                st.warning(f"Warning: Column '{col}' was not present in the original data and cannot be encoded.")

        return cleaned_data

    # Encode the user input data
    encoded_data = encode_categorical_data(user_data, label_encoders)

    # Make prediction
    y_pred = model.predict(encoded_data.values)
    predicted_rr = round(y_pred[0], 2)

    # Display result
    st.write(f"Predicted Recruitment Rate (RR): {predicted_rr}")
