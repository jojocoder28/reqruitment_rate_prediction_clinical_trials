# Recruitment Rate Prediction App

## Overview
This is a **Streamlit-based web application** designed to predict recruitment rates for clinical studies using two different LightGBM models. Users can input study-related details and receive a predicted recruitment rate based on the selected prediction method.

---

## Features
- **Two Prediction Methods**:
  - **Method 1**: Uses PCA (Principal Component Analysis) for dimensionality reduction on encoded features.
  - **Method 2**: Predicts directly without PCA.
- **User-Friendly Interface**: Intuitive forms for inputting study details.
- **Dynamic Encoding**: Handles categorical data and multi-label inputs like Conditions and Interventions.
- **Interactive Outputs**: Displays predicted recruitment rates in real-time.

---

## Prerequisites
### Software Requirements:
1. Python (>= 3.8)
2. Web Browser (any modern browser)

### Install Dependencies:
The required Python libraries are listed in the `requirements.txt` file. To install them, use the following command:

```bash
pip install -r requirements.txt
```

### Files Required:
Ensure the following files are in the project directory:
- `lightgbm_model_method1_2x2.joblib`: Pre-trained LightGBM model for Method 1.
- `lightgbm_model_method2.joblib`: Pre-trained LightGBM model for Method 2.
- `label_encoders_method1_2x2.pkl`: Label encoders for Method 1.
- `label_encoders_method2.joblib`: Label encoders for Method 2.
- `mlb_conditions.pkl`: MultiLabelBinarizer for Conditions in Method 1.
- `mlb_interventions.pkl`: MultiLabelBinarizer for Interventions in Method 1.
- `pca_conditions.pkl`: PCA model for Conditions in Method 1.
- `pca_interventions.pkl`: PCA model for Interventions in Method 1.
- `usecase_4_.xlsx`: The Excel file containing study-related data.

---

## How to Run the App
1. **Navigate to the Project Directory**:
   Open a terminal/command prompt and navigate to the folder containing the project files.

2. **Run the Streamlit App**:
   Use the following command:

   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**:
   The app will open in your default web browser. If not, copy and paste the provided URL (e.g., `http://localhost:8501`) into your browser.

---

## Using the App
1. **Choose a Prediction Method**:
   - Use the sidebar to select either `Method 1 - LightGBM (PCA 2x2)` or `Method 2 - LightGBM`.

2. **Input Study Details**:
   - Fill in the required fields in the form, such as Study Status, Conditions, Interventions, etc.

3. **Submit for Prediction**:
   - Click the `Predict Recruitment Rate` button.

4. **View Results**:
   - The predicted recruitment rate will be displayed on the screen.

---

## Troubleshooting
- **Missing Files**: Ensure all required files are in the same directory as the app.
- **Dependency Issues**: Reinstall dependencies using `pip install -r requirements.txt`.
- **Streamlit Issues**: Ensure Streamlit is installed correctly by running `pip install streamlit`.

---

## Notes for Beginners
- This app is built using **Streamlit**, a Python framework for building interactive web apps.
- If you're new to Python, install Python and the required libraries before running the app.
- Feel free to modify input forms and features as per your requirements!

---

## Contact
For any questions or feedback, please contact:
**Swarnadeep Das**
Email: dasjojo7@gmail.com

#
