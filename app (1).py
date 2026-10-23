"""Streamlit front end.  Run:  pip install streamlit scikit-learn pandas joblib
   python train_model.py && streamlit run app.py"""
import joblib, streamlit as st
from train_model import make_features

st.set_page_config(page_title="MPG Predictor", page_icon="⛽")
model = joblib.load("mpg_model.joblib")

st.title("MPG predictor")
st.caption("Gradient-boosted model trained on the auto-mpg dataset (1970–1982 cars).")
c1, c2 = st.columns(2)
cyl = c1.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
origin = c2.selectbox("Origin", [1, 2, 3], format_func={1: "USA", 2: "Europe", 3: "Japan"}.get)
disp = c1.number_input("Displacement (cu in)", 60.0, 500.0, 140.0)
hp = c2.number_input("Horsepower", 40.0, 250.0, 90.0)
wt = c1.number_input("Weight (lb)", 1000.0, 6000.0, 2500.0, step=50.0)
acc = c2.number_input("Acceleration (0–60 mph, s)", 6.0, 30.0, 15.0)
yr = st.slider("Model year", 1970, 1982, 1976)

if st.button("Predict MPG", type="primary"):
    mpg = float(model.predict(make_features([cyl], [disp], [hp], [wt], [acc], [yr - 1900], [origin]))[0])
    st.metric("Predicted fuel economy", f"{mpg:.1f} mpg")
    st.write(f"≈ {mpg * 0.425144:.1f} km/L  ·  {100 / (mpg * 0.425144):.1f} L/100 km")
