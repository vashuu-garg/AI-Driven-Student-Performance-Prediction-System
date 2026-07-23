import streamlit as st
import joblib
import numpy as np
import os


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# ---------------- LOAD MODEL ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "Model",
    "student_performance_model.pkl"
)

encoder_path = os.path.join(
    BASE_DIR,
    "Model",
    "encoders.pkl"
)


model = joblib.load(model_path)

encoders = joblib.load(encoder_path)



# ---------------- TITLE ----------------

st.title("🎓 AI-Driven Student Performance Prediction System")

st.write(
    "Enter student details below to predict the final performance score."
)



# ---------------- INPUTS ----------------


hours = st.number_input(
    "Hours Studied",
    min_value=0,
    max_value=24,
    value=5
)


attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)


parental_involvement = st.selectbox(
    "Parental Involvement",
    ["Low","Medium","High"]
)


resources = st.selectbox(
    "Access to Resources",
    ["Low","Medium","High"]
)


extra_activity = st.selectbox(
    "Extracurricular Activities",
    ["No","Yes"]
)


sleep = st.number_input(
    "Sleep Hours",
    min_value=0,
    max_value=24,
    value=7
)


previous_score = st.number_input(
    "Previous Scores",
    min_value=0,
    max_value=100,
    value=75
)


motivation = st.selectbox(
    "Motivation Level",
    ["Low","Medium","High"]
)


internet = st.selectbox(
    "Internet Access",
    ["No","Yes"]
)


tutoring = st.number_input(
    "Tutoring Sessions",
    min_value=0,
    max_value=10,
    value=2
)


family_income = st.selectbox(
    "Family Income",
    ["Low","Medium","High"]
)


teacher_quality = st.selectbox(
    "Teacher Quality",
    ["Low","Medium","High"]
)


school_type = st.selectbox(
    "School Type",
    ["Public","Private"]
)


peer = st.selectbox(
    "Peer Influence",
    ["Negative","Neutral","Positive"]
)


physical = st.number_input(
    "Physical Activity",
    min_value=0,
    max_value=10,
    value=3
)


learning = st.selectbox(
    "Learning Disabilities",
    ["No","Yes"]
)


education = st.selectbox(
    "Parental Education Level",
    ["High School","College","Postgraduate"]
)


distance = st.selectbox(
    "Distance From Home",
    ["Near","Moderate","Far"]
)


gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)



# ---------------- ENCODING FUNCTION ----------------


def encode_value(column, value):

    if column in encoders:
        return int(encoders[column].transform([value])[0])

    mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "No": 0,
        "Yes": 1,
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
        "Public": 0,
        "Private": 1,
        "Near": 0,
        "Moderate": 1,
        "Far": 2,
        "Male": 0,
        "Female": 1,
        "High School": 0,
        "College": 1,
        "Postgraduate": 2
    }

    return mapping[value]



# ---------------- PREDICTION ----------------


if st.button("Predict Performance 🚀"):


    input_data = [

        hours,
        attendance,

        encode_value(
            "Parental_Involvement",
            parental_involvement
        ),

        encode_value(
            "Access_to_Resources",
            resources
        ),

        encode_value(
            "Extracurricular_Activities",
            extra_activity
        ),

        sleep,
        previous_score,


        encode_value(
            "Motivation_Level",
            motivation
        ),


        encode_value(
            "Internet_Access",
            internet
        ),


        tutoring,


        encode_value(
            "Family_Income",
            family_income
        ),


        encode_value(
            "Teacher_Quality",
            teacher_quality
        ),


        encode_value(
            "School_Type",
            school_type
        ),


        encode_value(
            "Peer_Influence",
            peer
        ),


        physical,


        encode_value(
            "Learning_Disabilities",
            learning
        ),


        encode_value(
            "Parental_Education_Level",
            education
        ),


        encode_value(
            "Distance_from_Home",
            distance
        ),


        encode_value(
            "Gender",
            gender
        )

    ]


    input_array = np.array(input_data).reshape(1,-1)


    prediction = model.predict(input_array)



    score = round(float(prediction[0]),2)



    st.success(
        f"🎯 Predicted Final Score: {score}"
    )


    if score >= 85:
        st.info("Excellent Performance ⭐⭐⭐⭐⭐")

    elif score >= 70:
        st.info("Good Performance ⭐⭐⭐⭐")

    elif score >= 50:
        st.warning("Average Performance ⭐⭐⭐")

    else:
        st.error("Needs Improvement")
