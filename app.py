import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai

# Load model
model = pickle.load(open("model.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# Gemini API
genai.configure(api_key="")

st.title("AI Student Performance Advisor")

# Inputs
study = st.slider("Study Hours", 0.0, 12.0, 4.0)
attendance = st.slider("Attendance %", 0, 100, 75)
sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
screen = st.slider("Screen Time", 0.0, 15.0, 5.0)

if st.button("Predict"):

    # Create input dataframe
    input_dict = {
        "StudyHoursPerDay": study,
        "AttendancePercentage": attendance,
        "SleepHoursPerNight": sleep,
        "TotalScreenTime": screen
    }

    input_df = pd.DataFrame([input_dict])

    # Align columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    # Predict
    pred = model.predict(input_df)[0]

    st.subheader(f"Predicted Grade: {pred:.2f}")

    # Indicator
    if pred >= 8:
        st.success("Good Performance")
    elif pred >= 6:
        st.warning("Average Performance")
    else:
        st.error("At Risk")

    # Prompt
    prompt = f'''
    Predicted Grade: {pred:.2f}

    Important features:
    Study Hours: {study}
    Attendance: {attendance}
    Sleep: {sleep}
    Screen Time: {screen}

    Give 2-3 sentence friendly academic advice.
    '''

    # Gemini response
    try:
        model_gemini = genai.GenerativeModel("gemini-2.0-flash")
        response = model_gemini.generate_content(prompt)

        st.subheader("AI Recommendation")
        st.write(response.text)

    except Exception as e:
        
        st.error("LLM recommendation could not be generated.")
        st.write("Basic recommendation: Try improving attendance, keeping study hours consistent, and reducing excessive screen time.")
