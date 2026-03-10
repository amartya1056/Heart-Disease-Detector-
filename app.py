import streamlit as st
import numpy as np
import joblib
import google.generativeai as genai
from api_key import api_key
# Load the model
model = joblib.load('heart_disease_detection.pkl')

# Configure GenAI with API key
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

system_prompt = """
You are an advanced AI medical assistant specializing in cardiovascular health. Your primary role is to analyze patient health parameters and provide detailed explanations based on input data.

When a patient provides 13 medical inputs (age, sex, chest pain type, resting blood pressure, cholesterol level, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope of ST segment, number of major vessels, and thalassemia type), analyze these factors and generate a medical explanation based on known cardiovascular risk factors, medical studies, and diagnosis methods.

Your responses should be informative and based on the provided input values. Avoid generic advice and focus on explaining the specific risks, factors, and possible causes of heart disease for the given inputs.
"""

ai_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

st.title('Heart Disease Prediction System')

# Input fields
age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex (1 = Male, 0 = Female)', [0, 1])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
chol = st.number_input('Cholesterol Level', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', [0, 1])
restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=250, value=150)
exang = st.selectbox('Exercise-Induced Angina (1 = Yes, 0 = No)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels (0-4)', [0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (0-3)', [0, 1, 2, 3])

input_data = None  # Initialize input_data globally

# Prediction
if st.button('Predict'):
    input_data = np.array(
        [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success('You are Healthy')
    else:
        st.error('You have Heart Disease. Please consult a doctor.')

        # AI Medical Assistant Explanation
        user_query = f"Analyze the following patient data and explain the possible causes of heart disease: {input_data.tolist()}"
        response = ai_model.generate_content(user_query)

        st.write("### AI Medical Explanation:")
        st.write(response.text if response else "AI could not generate an explanation.")

# User Q&A Section
st.write("### Ask the Medical AI About Your Condition")
user_question = st.text_area("Enter your medical question about heart disease:")
if st.button("Get AI Response"):
    if user_question.strip():
        if input_data is not None:
            ai_response = ai_model.generate_content(
                f"Patient's provided data: {input_data.tolist()}. Now answer: {user_question}")
        else:
            ai_response = ai_model.generate_content(user_question)

        st.write("### AI Medical Response:")
        st.write(ai_response.text if ai_response else "AI could not generate a response.")
    else:
        st.warning("Please enter a question before clicking the button.")
