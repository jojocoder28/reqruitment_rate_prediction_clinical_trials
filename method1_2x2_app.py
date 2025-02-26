import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

st.title("Study Recruitment Rate Prediction")


# Load saved models and encoders
model = joblib.load('lightgbm_model_method1_2x2.joblib')
label_encoders = joblib.load('label_encoders_method1_2x2.pkl')
mlb_conditions = joblib.load('mlb_conditions.pkl')
mlb_interventions = joblib.load('mlb_interventions.pkl')
pca_conditions = joblib.load('pca_conditions.pkl')
pca_interventions = joblib.load('pca_interventions.pkl')

# Load your dataset for input options
data = pd.read_excel("./usecase_4_.xlsx")

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
        data[col] = label_encoders[col].transform(data[col].astype(str))  # Transform with the fitted encoder

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

# Input fields for user data

# Create a form with user input fields
with st.form(key='user_input_form'):
    allocation_options = data['Allocation'].dropna().unique()
    intervention_model_options = data['Intervention Model'].dropna().unique()
    primary_purpose_options = data['Primary Purpose'].dropna().unique()
    masking_options = data['Masking'].dropna().unique()
    
    study_status = st.selectbox('Study Status', data['Study Status'].unique())
    study_results = st.selectbox('Study Results', data['Study Results'].unique())
    sex = st.selectbox('Sex', data['Sex'].unique())
    age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1)
    phases = st.selectbox('Phases', data['Phases'].unique())
    enrollment = st.number_input('Enrollment', min_value=0, max_value=10000, value=100, step=1)
    funder_type = st.selectbox('Funder Type', data['Funder Type'].unique())
    study_type = st.selectbox('Study Type', data['Study Type'].unique())
    locations = st.selectbox('Locations', data['Locations'].unique())
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
        'Locations': [locations],
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
