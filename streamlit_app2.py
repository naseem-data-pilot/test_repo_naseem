import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import os

os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "False"


def load_model():
    with open(r'rf_modelsss.pkl', 'rb') as file:
        model = pickle.load(file)
    return model[0]
 
model = load_model()
 
def predict_risk(age, sex, housing, saving_accounts, checking_account, purpose):
    # Define mappings for categorical variables
    sex_mapping = {'Male': 0, 'Female': 1}
    housing_mapping = {'Own' : 0, 'Free':1, 'Rent': 2}
    saving_mapping = {'Little': 0, 'quite rich':1, 'Moderate': 2, 'Rich': 3}
    checking_mapping = {'Little': 0, 'Moderate': 1, 'Rich': 2}
    purpose_mapping = {'Radio/TV': 0, 'Education': 1, 'Furniture/Equipment': 2, 'Car': 3, 'Business': 4, 'Others': 5}
 
    # Map categorical inputs to numerical values
    sex = sex_mapping[sex]
    housing = housing_mapping[housing]
    saving_accounts = saving_mapping[saving_accounts]
    checking_account = checking_mapping[checking_account]
    purpose = purpose_mapping[purpose]
 
    # Create input array
    input_array = np.array([[age, sex, housing, saving_accounts, checking_account, purpose]])
   
    # Make prediction
    prediction = model.predict(input_array)[0]  # For binary classification
    return prediction
 
def show_predict_page():
    st.title('Credit Risk Prediction App')
    st.write("""### Enter information to predict credit risk""")
 
    sex_options = ['Male', 'Female']
    housing_options = ['Own', 'Free', 'Rent']
    saving_accounts_options = ['Little','quite rich', 'Moderate', 'Rich']
    checking_account_options = ['Little', 'Moderate', 'Rich']
    purpose_options = ['Radio/TV', 'Education', 'Furniture/Equipment', 'Car', 'Business', 'Others']
 
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    sex = st.selectbox('Sex', sex_options)
    housing = st.selectbox('Housing', housing_options)
    saving_accounts = st.selectbox('Saving accounts', saving_accounts_options)
    checking_account = st.selectbox('Checking account', checking_account_options)
    purpose = st.selectbox('Purpose', purpose_options, index=1)
 
    ok = st.button('Predict Risk')
    if ok:
        result = predict_risk(age, sex, housing, saving_accounts, checking_account, purpose)
       
        # Display result in a pie chart using Plotly
        labels = ['Low Risk', 'High Risk']
        values = [1 - result, result]  # Assuming result is the probability of high risk
        fig = px.pie(names=labels, values=values, title='Credit Risk Prediction')
        st.plotly_chart(fig)
        st.subheader(f"The predicted risk is: {result}")
 
# Usage
show_predict_page()
