import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="AI Health Predictor",
    page_icon="🩺",
    layout="centered"
)

# -------------------------
# Simple styling
# -------------------------
st.markdown(
    """
    <style>
      .result-card {
        padding: 1.1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        margin-top: 0.75rem;
      }
      .muted {
        opacity: 0.8;
        font-size: 0.95rem;
      }
      .tiny {
        opacity: 0.7;
        font-size: 0.85rem;
      }
      .pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid rgba(255,255,255,0.15);
      }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Load model + scaler
# -------------------------
@st.cache_resource
def load_artifacts():
    with open("models/baseline_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_artifacts()

# -------------------------
# Header
# -------------------------
st.title("🩺 AI Health Predictor")
st.write("Enter health parameters to estimate **diabetes risk probability** using a trained ML model.")
st.info("⚠️ This is a demo prediction tool. It is **not medical advice** or a diagnosis.", icon="ℹ️")

# -------------------------
# Sidebar inputs (form)
# -------------------------
st.sidebar.header("Patient Inputs")

with st.sidebar.form("input_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)

    glucose = st.slider("Glucose", min_value=0, max_value=250, value=120, step=1)
    blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=140, value=70, step=1)
    skin_thickness = st.slider("Skin Thickness", min_value=0, max_value=99, value=20, step=1)

    insulin = st.slider("Insulin", min_value=0, max_value=850, value=120, step=1)
    bmi = st.slider("BMI", min_value=0.0, max_value=67.0, value=32.0, step=0.1)

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=2.5, value=0.50, step=0.01,
        help="Family history influence factor (dataset-defined)."
    )

    age = st.slider("Age", min_value=1, max_value=120, value=30, step=1)

    submitted = st.form_submit_button("Predict Risk")

# -------------------------
# Build input dict
# -------------------------
input_data = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
}

# -------------------------
# Prediction logic (on submit)
# -------------------------
if submitted:
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    proba = model.predict_proba(input_scaled)[0, 1]
    pred = int(proba >= 0.5)

    # Risk banding (simple + readable)
    if proba < 0.33:
        risk_level = "Low"
        status = st.success
        pill_color = "#2ecc71"
        guidance = [
            "Risk appears low based on the inputs.",
            "Keep healthy habits and schedule regular checkups."
        ]
    elif proba < 0.66:
        risk_level = "Medium"
        status = st.warning
        pill_color = "#f1c40f"
        guidance = [
            "Risk appears moderate based on the inputs.",
            "Consider lifestyle improvements and consult a clinician if concerned."
        ]
    else:
        risk_level = "High"
        status = st.error
        pill_color = "#e74c3c"
        guidance = [
            "Risk appears high based on the inputs.",
            "It may be worth discussing screening/testing with a medical professional."
        ]

    # Result card
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("Results")
    st.write(
        f"**Diabetes Risk Probability:** `{proba:.2f}`  "
        f"({proba*100:.1f}%)"
    )

    st.markdown(
        f'<span class="pill" style="background:{pill_color}22; border-color:{pill_color}55;">Risk Level: {risk_level}</span>',
        unsafe_allow_html=True
    )

    st.write("")
    st.progress(int(proba * 100))

    status(" ".join(guidance))

    st.markdown(
        "<p class='tiny'>Tip: The probability is based on a machine learning model trained on an open dataset. "
        "It may be wrong for individual cases.</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

   
    with st.expander("See input values used for this prediction"):
        st.dataframe(input_df)

# -------------------------
# Explainability section
# -------------------------
st.write("")
st.subheader("🧠 Explainability")

with st.expander("How this prediction is made (simple explanation)"):
    st.markdown(
        """
**Model used:** Logistic Regression (Balanced)

**What happens when you click Predict:**
1. Your inputs are placed into a single-row table (DataFrame) with 8 features.
2. The app applies the same **StandardScaler** used during training (so the model sees data in the format it expects).
3. The model outputs a **probability** between 0 and 1 for the positive class (higher diabetes risk).
4. The app converts that probability into a simple **risk level**:
   - Low: < 0.33  
   - Medium: 0.33–0.66  
   - High: > 0.66

**Why we chose this model:**  
On the test set, it performed better overall and had **much higher recall** than the deep learning model (meaning it misses fewer positive cases).
        """
    )

with st.expander("What each input generally represents"):
    st.markdown(
        """
- **Pregnancies:** Number of pregnancies (dataset feature).
- **Glucose:** Plasma glucose concentration (higher often correlates with risk).
- **BloodPressure:** Diastolic blood pressure (mm Hg).
- **SkinThickness:** Triceps skin fold thickness (mm).
- **Insulin:** 2-Hour serum insulin (mu U/ml).
- **BMI:** Body Mass Index (weight vs height).
- **DiabetesPedigreeFunction:** A dataset-defined score for family history influence.
- **Age:** Age in years.
        """
    )

# -------------------------
# Footer
# -------------------------
st.write("")
st.markdown("<p class='tiny'>Built with Streamlit • Model: Logistic Regression (Balanced) • Dataset: diabetes (open-source)</p>", unsafe_allow_html=True)
