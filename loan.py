import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------------------
# Custom CSS - dark, card-style inputs
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e0e10;
    }
    section.main > div {
        max-width: 1200px;
    }
    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
    }
    h2 {
        font-weight: 800 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-size: 1rem !important;
        color: #d0d0d5 !important;
        margin-bottom: 0.3rem !important;
    }
    div[data-baseweb="input"], div[data-baseweb="select"] {
        border-radius: 10px !important;
    }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1c1c1f !important;
        border-radius: 10px !important;
        border: 1px solid #2c2c30 !important;
        color: #f5f5f5 !important;
    }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        padding: 0.7rem 0;
        background-color: #f0a500;
        color: #0e0e10;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #ffb81c;
        color: #0e0e10;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Load the trained pipeline (preprocessor + XGBClassifier)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("best_res.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

DEFAULT_THRESHOLD = 0.5  # standard threshold — matches the 95.8% accuracy reported in the notebook

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("# 🏦 AI-Powered Loan Approval")


# ---------------------------------------------------------------
# Financial Information
# ---------------------------------------------------------------
st.markdown("## Financial Information")

c1, c2 = st.columns(2)
with c1:
    Applicant_Income = st.number_input("Applicant Income", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
with c2:
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0, value=0.0, step=1000.0, format="%.2f")

c1, c2 = st.columns(2)
with c1:
    Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
with c2:
    Collateral_Value = st.number_input("Collateral Value", min_value=0.0, value=0.0, step=1000.0, format="%.2f")

c1, c2 = st.columns(2)
with c1:
    Savings = st.number_input("Savings", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
with c2:
    Existing_Loans = st.number_input("Existing Loans", min_value=0, value=0, step=1)

c1, c2 = st.columns(2)
with c1:
    Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900, value=700, step=1)
with c2:
    Loan_Term = st.number_input("Loan Term (Months)", min_value=1, max_value=480, value=120, step=1)

DTI_Ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

# ---------------------------------------------------------------
# Personal Information
# ---------------------------------------------------------------
st.markdown("## Personal Information")

c1, c2 = st.columns(2)
with c1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
with c2:
    Education_Level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

c1, c2 = st.columns(2)
with c1:
    Dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0, step=1)
with c2:
    Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])

c1, c2 = st.columns(2)
with c1:
    Gender = st.selectbox("Gender", ["Female", "Male"])
with c2:
    Employer_Category = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Business", "Unemployed"])

Marital_Status = st.selectbox("Marital Status", ["Married", "Single"])

# ---------------------------------------------------------------
# Loan Details
# ---------------------------------------------------------------
st.markdown("## Loan Details")

c1, c2 = st.columns(2)
with c1:
    Loan_Purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Personal", "Business", "Education"])
with c2:
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

threshold = DEFAULT_THRESHOLD  # fixed at 0.5 (standard cutoff, matches 95.8% test accuracy)

# ---------------------------------------------------------------
# Build input dataframe in the EXACT column order used for training
# ---------------------------------------------------------------
input_dict = {
    "Applicant_Income": Applicant_Income,
    "Coapplicant_Income": Coapplicant_Income,
    "Employment_Status": Employment_Status,
    "Age": Age,
    "Marital_Status": Marital_Status,
    "Dependents": Dependents,
    "Credit_Score": Credit_Score,
    "Existing_Loans": Existing_Loans,
    "DTI_Ratio": DTI_Ratio,
    "Savings": Savings,
    "Collateral_Value": Collateral_Value,
    "Loan_Amount": Loan_Amount,
    "Loan_Term": Loan_Term,
    "Loan_Purpose": Loan_Purpose,
    "Property_Area": Property_Area,
    "Education_Level": Education_Level,
    "Gender": Gender,
    "Employer_Category": Employer_Category,
}
input_df = pd.DataFrame([input_dict])

with st.expander("🔍 View input data"):
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

# ---------------------------------------------------------------
# Predict
# ---------------------------------------------------------------
if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
    try:
        proba_approve = float(model.predict_proba(input_df)[0][1])
        proba_reject = float(1 - proba_approve)
        prediction = int(proba_approve >= threshold)

        st.markdown("## Result")

        if prediction == 1:
            st.success("### ✅ Loan Approved")
        else:
            st.error("### ❌ Loan Rejected")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Probability of Approval", f"{proba_approve*100:.2f}%")
            st.progress(min(max(proba_approve, 0.0), 1.0))
        with col2:
            st.metric("❌ Probability of Rejection", f"{proba_reject*100:.2f}%")
            st.progress(min(max(proba_reject, 0.0), 1.0))

        st.caption("Prediction made using a decision threshold of 0.5 on the predicted probability of approval.")

    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")