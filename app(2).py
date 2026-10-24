
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from train_model import make_features

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Car Mileage Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mpg_model.joblib"

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path.name}. "
            "Please add mpg_model.joblib to your repository."
        )

    return joblib.load(model_path)


st.title("🚗 Car Mileage Predictor")
st.caption(
    "Estimate fuel economy using your car's specifications."
)

st.markdown(
    """
    Enter your car's details below to estimate its fuel economy.
    This application uses a trained machine learning model.
    """
)

# --------------------------------------------------
# Load model safely
# --------------------------------------------------
try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()

# --------------------------------------------------
# Input form
# --------------------------------------------------
with st.form("car_details_form"):

    st.subheader("Car Specifications")

    col1, col2 = st.columns(2)

    with col1:
        cylinders = st.selectbox(
            "Cylinders",
            options=[3, 4, 5, 6, 8],
            index=1,
            help="Number of engine cylinders."
        )

        displacement = st.number_input(
            "Displacement (cu in)",
            min_value=60.0,
            max_value=500.0,
            value=140.0,
            step=5.0
        )

        horsepower = st.number_input(
            "Horsepower",
            min_value=40.0,
            max_value=250.0,
            value=90.0,
            step=5.0
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
            value=15.0,
            step=0.5
        )

        origin = st.selectbox(
            "Origin",
            options=[1, 2, 3],
            format_func=lambda value: {
                1: "USA",
                2: "Europe",
                3: "Japan"
            }[value]
        )

    year = st.slider(
        "Model Year",
        min_value=1970,
        max_value=1982,
        value=1976
    )

    submitted = st.form_submit_button(
        "🚀 Predict MPG",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# Make prediction
# --------------------------------------------------
if submitted:
    try:
        # Use the same feature-engineering function
        # that was used when training the model.
        features = make_features(
            cylinders,
            displacement,
            horsepower,
            weight,
            acceleration,
            year,
            origin
        )

        # Convert to DataFrame only if appropriate for
        # the feature format expected by your model.
        if not isinstance(features, (pd.DataFrame,)):
            # Keep the feature structure returned by
            # make_features unless it needs named columns.
            pass

        prediction = float(model.predict(features)[0])

        if not pd.notna(prediction) or prediction <= 0:
            st.error(
                "The model returned an invalid MPG prediction. "
                "Please check your trained model and input features."
            )
            st.stop()

        # Convert US MPG to litres per 100 kilometres.
        liters_per_100_km = 235.215 / prediction

        st.divider()
        st.subheader("Prediction Results")

        st.success("Prediction completed successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Estimated Fuel Economy",
                value=f"{prediction:.2f} MPG"
            )

        with col2:
            st.metric(
                label="Fuel Consumption",
                value=f"{liters_per_100_km:.2f} L/100 km"
            )

        st.caption(
            "Higher MPG means better fuel efficiency. "
            "Lower L/100 km means less fuel consumed."
        )

        with st.expander("View submitted car specifications"):
            st.write({
                "Cylinders": cylinders,
                "Displacement (cu in)": displacement,
                "Horsepower": horsepower,
                "Weight (lb)": weight,
                "Acceleration (seconds)": acceleration,
                "Model Year": year,
                "Origin": {
                    1: "USA",
                    2: "Europe",
                    3: "Japan"
                }[origin]
            })

        st.info(
            "This result is an estimate from a machine learning model, "
            "not a guarantee of real-world fuel economy."
        )

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)
