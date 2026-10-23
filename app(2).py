
import streamlit as st
import joblib
from train_model import make_features

st.set_page_config(
    page_title="MPG Predictor",
    page_icon="🚗",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("mpg_model.joblib")

model = load_model()

st.title("🚗 Car Mileage Predictor")
st.caption(
    "Predict fuel economy using a car's specifications."
)

st.write(
    "Enter the details below to estimate your car's mileage."
)

col1, col2 = st.columns(2)

with col1:
    cylinders = st.selectbox(
        "Cylinders", [3, 4, 5, 6, 8], index=1
    )

    displacement = st.number_input(
        "Displacement (cu in)",
        min_value=60.0,
        max_value=500.0,
        value=140.0
    )

    horsepower = st.number_input(
        "Horsepower",
        min_value=40.0,
        max_value=250.0,
        value=90.0
    )

with col2:
    weight = st.number_input(
        "Weight (lb)",
        min_value=1000.0,
        max_value=6000.0,
        value=2500.0,
        step=50.0
    )

    acceleration = st.number_input(
        "Acceleration (0–60 mph, seconds)",
        min_value=6.0,
        max_value=30.0,
        value=15.0
    )

    origin = st.selectbox(
        "Origin",
        [1, 2, 3],
        format_func=lambda x: {
            1: "USA",
            2: "Europe",
            3: "Japan"
        }[x]
    )

year = st.slider(
    "Model year",
    min_value=1970,
    max_value=1982,
    value=1976
)

if st.button("Predict MPG", type="primary"):
    try:
        features = make_features(
            cylinders,
            displacement,
            horsepower,
            weight,
            acceleration,
            year,
            origin
        )

        prediction = float(model.predict(features)[0])

        st.success(f"Predicted fuel economy: {prediction:.2f} MPG")

        st.metric(
            "Estimated mileage",
            f"{prediction:.2f} MPG"
        )

        st.write(
            f"Equivalent fuel consumption: "
            f"{235.215 / prediction:.2f} L/100 km"
            if prediction > 0
            else "Fuel consumption conversion is unavailable."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
