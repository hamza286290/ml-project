import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── 4.2 Page Title ──
st.set_page_config(
    page_title="CTG Fetal State Classifier",
    page_icon="🫀",
    layout="wide"
)


# ── Custom CSS ──
# st.markdown("""
#     <style>
#     .title { font-size: 2.5rem; font-weight: 800; color: #4CAF50; }
#     .subtitle { font-size: 1rem; opacity: 0.7; margin-bottom: 2rem; }
#     .result-normal { background-color: #1e5631; padding: 1rem; border-radius: 8px; color: #ffffff; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem; }
#     .result-suspect { background-color: #7d6608; padding: 1rem; border-radius: 8px; color: #ffffff; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem; }
#     .result-pathologic { background-color: #7b1c1c; padding: 1rem; border-radius: 8px; color: #ffffff; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem; }
#     </style>
# """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .result-normal { background-color: #1e5631; padding: 1rem; border-radius: 8px; color: #ffffff; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem; }
    .result-suspect { background-color: #7d6608; padding: 1rem; border-radius: 8px; color: #ffffff; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem; }
    .result-pathologic { background-color: #7b1c1c; padding: 1rem; border-radius: 8px; color: #ffffff; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem; }
    div.stButton > button { background-color: #ff4b4b; color: white; font-weight: bold; border: none; border-radius: 8px; }
    div.stButton > button:hover { background-color: #cc0000; color: white; }
    </style>
""", unsafe_allow_html=True)

# ── 4.2 App Title ──
st.title("🫀 CTG Fetal State Classifier")
st.markdown("Cardiotocography (CTG) based fetal state classification — Normal, Suspect, or Pathologic")


# ── 4.3 Load Models + Scaler ──
@st.cache_resource
def load_models():
    rf     = joblib.load('rf_best_model.pkl')
    svm    = joblib.load('svm_best_model.pkl')
    lr     = joblib.load('lr_best_model.pkl')
    scaler = joblib.load('scaler.pkl')          # ← load scaler
    return rf, svm, lr, scaler

try:
    rf_model, svm_model, lr_model, scaler = load_models()
    st.success("✅ Models and scaler loaded successfully!")
except:
    st.error("❌ Files not found. Please ensure rf_best_model.pkl, svm_best_model.pkl, lr_best_model.pkl and scaler.pkl are in the same directory.")
    st.stop()

# ── 4.4 Get Input via Streamlit Input Functions ──
st.markdown("---")
st.subheader("📋 Enter Patient CTG Features")
st.markdown("Adjust the sliders below to enter the patient's CTG readings:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**FHR Signal Features**")
    LB       = st.slider("LB — FHR Baseline (bpm)",        min_value=106,  max_value=160,  value=132,  step=1)
    AC       = st.slider("AC — Accelerations/sec",          min_value=0,    max_value=26,   value=3,    step=1)
    FM       = st.slider("FM — Fetal Movements/sec",        min_value=0,    max_value=564,  value=0,    step=1)
    UC       = st.slider("UC — Uterine Contractions/sec",   min_value=0,    max_value=23,   value=2,    step=1)
    ASTV     = st.slider("ASTV — % Abnormal STV",           min_value=12,   max_value=87,   value=20,   step=1)
    MSTV     = st.slider("MSTV — Mean STV",                 min_value=0.2,  max_value=7.0,  value=2.0,  step=0.1)
    ALTV     = st.slider("ALTV — % Abnormal LTV",           min_value=0,    max_value=91,   value=8,    step=1)
    MLTV     = st.slider("MLTV — Mean LTV",                 min_value=0.0,  max_value=50.7, value=12.0, step=0.1)

with col2:
    st.markdown("**Deceleration Features**")
    DL       = st.slider("DL — Light Decelerations",        min_value=0,    max_value=16,   value=0,    step=1)
    DS       = st.slider("DS — Severe Decelerations",       min_value=0,    max_value=1,    value=0,    step=1)
    DP       = st.slider("DP — Prolonged Decelerations",    min_value=0,    max_value=4,    value=0,    step=1)

    st.markdown("**Histogram Features**")
    Width    = st.slider("Width — FHR Histogram Width",     min_value=3,    max_value=180,  value=100,  step=1)
    Min      = st.slider("Min — Minimum FHR",               min_value=50,   max_value=159,  value=120,  step=1)
    Max      = st.slider("Max — Maximum FHR",               min_value=122,  max_value=238,  value=160,  step=1)
    Nmax     = st.slider("Nmax — Number of Peaks",          min_value=0,    max_value=18,   value=6,    step=1)
    Nzeros   = st.slider("Nzeros — Number of Zeros",        min_value=0,    max_value=10,   value=1,    step=1)

with col3:
    st.markdown("**Statistical Features**")
    Mode     = st.slider("Mode — FHR Mode",                 min_value=60,   max_value=187,  value=140,  step=1)
    Mean     = st.slider("Mean — Mean FHR",                 min_value=73,   max_value=182,  value=138,  step=1)
    Median   = st.slider("Median — Median FHR",             min_value=77,   max_value=186,  value=139,  step=1)
    Variance = st.slider("Variance — FHR Variance",         min_value=0,    max_value=269,  value=10,   step=1)
    Tendency = st.slider("Tendency — FHR Trend",            min_value=-1,   max_value=1,    value=0,    step=1)

    st.markdown("**Select Model**")
    model_choice = st.selectbox(
        "Choose Model:",
        ["Random Forest", "SVM", "Logistic Regression", "All Models"]
    )

# ── 4.5 Make Predictions ──
st.markdown("---")

# Collect input
feature_names = ['LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                 'DL', 'DS', 'DP', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
                 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']

input_data = pd.DataFrame([[
    LB, AC, FM, UC, ASTV, MSTV, ALTV, MLTV,
    DL, DS, DP, Width, Min, Max, Nmax, Nzeros,
    Mode, Mean, Median, Variance, Tendency
]], columns=feature_names)

label_map = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}
emoji_map = {'Normal': '✅', 'Suspect': '⚠️', 'Pathologic': '🚨'}

def display_result(name, prediction):
    label = label_map[int(prediction)]
    emoji = emoji_map[label]
    css_class = f"result-{label.lower()}"
    st.markdown(
        f'<div class="{css_class}">{emoji} {name}: <strong>{label}</strong></div>',
        unsafe_allow_html=True
    )
    st.markdown("")

if st.button("🔍 Predict Fetal State", use_container_width=True):

    # Scale input for SVM and LR
    input_scaled = scaler.transform(input_data)  # ← scale using loaded scaler

    st.subheader("🎯 Prediction Results")

    if model_choice == "Random Forest":
        pred = rf_model.predict(input_data)        # unscaled ✅
        display_result("Random Forest", pred[0])

    elif model_choice == "SVM":
        pred = svm_model.predict(input_scaled)     # scaled ✅
        display_result("SVM", pred[0])

    elif model_choice == "Logistic Regression":
        pred = lr_model.predict(input_scaled)      # scaled ✅
        display_result("Logistic Regression", pred[0])

    elif model_choice == "All Models":
        rf_pred  = rf_model.predict(input_data)    # unscaled ✅
        svm_pred = svm_model.predict(input_scaled) # scaled ✅
        lr_pred  = lr_model.predict(input_scaled)  # scaled ✅

        display_result("Random Forest",       rf_pred[0])
        display_result("SVM",                 svm_pred[0])
        display_result("Logistic Regression", lr_pred[0])

    # Show input summary
    st.markdown("---")
    st.subheader("📊 Input Summary")
    st.dataframe(input_data, use_container_width=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.8rem;'>
    CTG Fetal State Classifier | Cardiotocography Dataset — UCI ML Repository
    </div>
""", unsafe_allow_html=True)
