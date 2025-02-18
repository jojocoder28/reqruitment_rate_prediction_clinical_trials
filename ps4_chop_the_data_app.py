from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Load the pre-trained models and encoders for both pages
model_method1 = load("lightgbm_model_method1_2x2.joblib")
model_method2 = load("lightgbm_model_method2.joblib")
label_encoders_method1 = load('label_encoders_method1_2x2.pkl')
label_encoders_method2 = load('label_encoders_method2.joblib')
mlb_conditions_method1 = load('mlb_conditions.pkl')
mlb_interventions_method1 = load('mlb_interventions.pkl')
pca_conditions_method1 = load('pca_conditions.pkl')
pca_interventions_method1 = load('pca_interventions.pkl')

# Load the data (replace with the actual path to the Excel file)
data = pd.read_excel("./usecase_4_.xlsx")


model = load('lightgbm_model_method1_2x2.joblib')
label_encoders = load('label_encoders_method1_2x2.pkl')
mlb_conditions = load('mlb_conditions.pkl')
mlb_interventions = load('mlb_interventions.pkl')
pca_conditions = load('pca_conditions.pkl')
pca_interventions = load('pca_interventions.pkl')


# Feature engineering function
def feature_engineering(data):
    data['Sex'].fillna(data['Sex'].mode()[0], inplace=True)
    data['Completion Date'].fillna(data['Completion Date'].mode()[0], inplace=True)
    data['Secondary Outcome Measures'].fillna("Not Specified", inplace=True)
    date_columns = ['Start Date', 'Primary Completion Date', 'Completion Date', 'Last Update Posted']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    
    study_design_split = data['Study Design'].str.split('|', expand=True)
    data['Allocation'] = study_design_split[0]
    data['Intervention Model'] = study_design_split[1]
    data['Primary Purpose'] = study_design_split[2]
    data['Masking'] = study_design_split[3]

    data['Start_Year'] = data['Start Date'].dt.year
    data['Start_Month'] = data['Start Date'].dt.month
    data['Study_Duration'] = (data['Completion Date'] - data['Start Date']).dt.days
    data['Late_Study'] = (data['Completion Date'] - data['Primary Completion Date']).dt.days
    data['Days_Since_Started'] = (pd.Timestamp.now() - data['Start Date']).dt.days
    data['Start_Season'] = data['Start Date'].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
    return data

# Preprocessing function
def preprocess_data(data, label_encoders, mlb_conditions, mlb_interventions, pca_conditions, pca_interventions):
    # Multi-hot encoding for Conditions (only for user input)
    data['Conditions'] = data['Conditions'].str.split('|')
    conditions_encoded = mlb_conditions.transform(data['Conditions'])
    conditions_pca = pca_conditions.transform(conditions_encoded)
    conditions_pca_df = pd.DataFrame(conditions_pca, columns=[f'PC_Conditions_{i+1}' for i in range(conditions_pca.shape[1])])
    data = pd.concat([data, conditions_pca_df], axis=1).drop('Conditions', axis=1)

    # Multi-hot encoding for Interventions (only for user input)
    data['Interventions'] = data['Interventions'].str.split('|')
    interventions_encoded = mlb_interventions.transform(data['Interventions'])
    interventions_pca = pca_interventions.transform(interventions_encoded)
    interventions_pca_df = pd.DataFrame(interventions_pca, columns=[f'PC_Interventions_{i+1}' for i in range(interventions_pca.shape[1])])
    data = pd.concat([data, interventions_pca_df], axis=1).drop('Interventions', axis=1)

    # Transform categorical columns using the fitted LabelEncoders
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    le_classes = set(le.classes_)
                    new_classes = set(data[col].unique()) - le_classes

                    if new_classes:
                        le.classes_ = np.append(le.classes_, list(new_classes))

                    data[col] = le.transform(data[col])
                else:
                    st.warning(f"Warning: Column '{col}' was not present in the original data and cannot be encoded.")  # Transform with the fitted encoder

    # Standardize numeric columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    return X_scaled

# Load the data to fit LabelEncoders
data = feature_engineering(data)
df = data.copy()
categorical_columns = data.select_dtypes(include=['object']).columns

# Fit LabelEncoders on the loaded data
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Fit the encoder and transform the data
    label_encoders[col] = le  # Store the encoder for future use







# Feature engineering for both models
study_design_split = data['Study Design'].str.split('|', expand=True)
data['Allocation'] = study_design_split[0]
data['Intervention Model'] = study_design_split[1]
data['Primary Purpose'] = study_design_split[2]
data['Masking'] = study_design_split[3]

data = data.drop(columns=['Study Design'])

# Streamlit app interface
st.title('TrialFast AI by Chop The Data')

# Sidebar for page navigation
page = st.sidebar.radio("Select a page", ["Method 1 - LightGBM (PCA 2x2)", "Method 2 - LightGBM"])

# Method 2 - LightGBM Prediction
if page == "Method 2 - LightGBM":
    st.header('Study Recruitment Rate Prediction - Method 2')

    # Form for user inputs for Method 2
    with st.form(key='prediction_form_method1'):
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

        # User input fields for Method 2
        study_status = st.selectbox('Study Status', study_status_options)
        study_results = st.selectbox('Study Results', study_results_options)
        conditions = st.multiselect('Conditions', conditions_data)
        interventions = st.multiselect('Interventions', interventions_data)
        sex = st.selectbox('Sex', sex_options)
        # age = st.number_input('Age', min_value=18, max_value=100, step=1)
        age = st.selectbox('Age',data['Age'].dropna().unique())
        phases = st.selectbox('Phases', data['Phases'].dropna().unique())
        enrollment = st.number_input('Enrollment', min_value=1, max_value=10000, step=1)
        funder_type = st.selectbox('Funder Type', funder_type_options)
        study_type = st.selectbox('Study Type', study_type_options)
        locations = st.multiselect('Locations', data['Locations'].str.split('|').explode().unique())
        study_duration = st.number_input('Study Duration (days)', min_value=1, max_value=10000, step=1)
        allocation = st.selectbox('Allocation', allocation_options)
        intervention_model = st.selectbox('Intervention Model', intervention_model_options)
        primary_purpose = st.selectbox('Primary Purpose', primary_purpose_options)
        masking = st.selectbox('Masking', masking_options)

        # Submit button
        submit_button = st.form_submit_button(label='Predict Recruitment Rate')

    if submit_button:
        # Prepare the user input data for Method 2
        user_data_method2 = pd.DataFrame({
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
            'Locations': ['|'.join(locations)],
            'Study_Duration': [study_duration],
            'Allocation': [allocation],
            'Intervention Model': [intervention_model],
            'Primary Purpose': [primary_purpose],
            'Masking': [masking]
        })

        # Encode the user input data for Method 2
        def encode_categorical_data_method2(data, label_encoders):
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

        # Encode the user input data for Method 1
        encoded_data_method2 = encode_categorical_data_method2(user_data_method2, label_encoders_method2)

        # Make prediction using Method 1's model
        prediction_method2 = model_method2.predict(encoded_data_method2.values)
        predicted_rr_method2 = round(prediction_method2[0], 2)

        # Display result
        st.write(f"Predicted Recruitment Rate (RR) using Method 2: {predicted_rr_method2}")

# Method 2 - LightGBM Prediction
elif page == "Method 1 - LightGBM (PCA 2x2)":
    st.header('Study Recruitment Rate Prediction - Method 1')
    st.markdown(
    '<p style="color: white; background-color: red; padding: 10px; font-size: 16px; font-weight: bold;">⚠️ WARNING: This method is unreliable! Use Method 2 for accurate results! ⚠️</p>',
    unsafe_allow_html=True)

    # Form for user inputs for Method 1
    with st.form(key='user_input_form'):
        allocation_options = data['Allocation'].dropna().unique()
        intervention_model_options = data['Intervention Model'].dropna().unique()
        primary_purpose_options = data['Primary Purpose'].dropna().unique()
        masking_options = data['Masking'].dropna().unique()
        
        study_status = st.selectbox('Study Status', data['Study Status'].unique())
        study_results = st.selectbox('Study Results', data['Study Results'].unique())
        sex = st.selectbox('Sex', data['Sex'].unique())
        # age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1)
        age = st.selectbox('Age',data['Age'].dropna().unique())
        phases = st.selectbox('Phases', data['Phases'].unique())
        enrollment = st.number_input('Enrollment', min_value=0, max_value=10000, value=100, step=1)
        funder_type = st.selectbox('Funder Type', data['Funder Type'].unique())
        study_type = st.selectbox('Study Type', data['Study Type'].unique())
        # locations = st.selectbox('Locations', data['Locations'].unique())
        locations = st.multiselect('Locations', data['Locations'].str.split('|').explode().unique())
        study_duration = st.number_input('Study Duration (Days)', min_value=1, max_value=10000, value=100, step=1)
        allocation = st.selectbox('Allocation', allocation_options)
        intervention_model = st.selectbox('Intervention Model', intervention_model_options)
        primary_purpose = st.selectbox('Primary Purpose', primary_purpose_options)
        masking = st.selectbox('Masking', masking_options)
        
        # Input fields for Conditions and Interventions (multi-select)
        conditions = st.multiselect('Conditions', data['Conditions'].str.split('|').explode().unique())
        interventions = st.multiselect('Interventions', data['Interventions'].str.split('|').explode().unique())

        # Submit button for prediction
        submit_button = st.form_submit_button(label="Predict Recruitment Rate")

    # If the button is pressed, process the input and make a prediction
    if submit_button:
        user_data = {
            'Study Status': [study_status],
            'Study Results': [study_results],
            'Sex': [sex],
            'Age': [age],
            'Phases': [phases],
            'Enrollment': [enrollment],
            'Funder Type': [funder_type],
            'Study Type': [study_type],
            'Locations': ['|'.join(locations)],
            'Study_Duration': [study_duration],
            'Allocation': [allocation],
            'Intervention Model': [intervention_model],
            'Primary Purpose': [primary_purpose],
            'Masking': [masking],
            'Conditions': ['|'.join(conditions)],
            'Interventions': ['|'.join(interventions)]
        }

        user_input_df = pd.DataFrame(user_data)

        # Preprocess the user input data using the saved encoders (and not the loaded data)
        user_input_data = preprocess_data(user_input_df, label_encoders, mlb_conditions, mlb_interventions, pca_conditions, pca_interventions)

        # Display the user input data for confirmation
        # st.write("### User Input Data")
        # st.write(user_data)

        # Predict using the model
        prediction = model.predict(user_input_data)

        # Display prediction result
        st.write(f"### Predicted Study Recruitment Rate: {prediction[0]:.2f}")
