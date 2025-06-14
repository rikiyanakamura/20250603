import streamlit as st
import pandas as pd
import pickle
import numpy as np

# モデルとエンコーダーを読み込む（pickle形式）
@st.cache_resource
def load_model():
    with open("rf_model_lnmeta_cloud.pkl", "rb") as f:
        model, features, label_encoders = pickle.load(f)
    return model, features, label_encoders

model, features, label_encoders = load_model()

st.title("Lymph Node Metastasis Prediction (Ln meta.)")

# 入力フォーム
age = st.number_input("Age", 20, 100)
height = st.number_input("Height (cm)", 130, 200)
weight = st.number_input("Weight (kg)", 30, 150)
diagnostic = st.selectbox("Axillary Diagnostic", ["Imaging", "FNA negative", "CNB negative"])
menopause = st.selectbox("Menopause", ["Pre", "Post"])
ct = st.selectbox("Clinical T stage", ["T1", "T2", "T3"])
histology = st.selectbox("CNB Histopathology", ["IDC", "ILC", "DCIS", "Other"])
chg = st.selectbox("Clinical Grade (cHG)", ["1", "2", "3"])
cer = st.slider("cER (%)", 0, 100)
cpgr = st.slider("cPgR (%)", 0, 100)
cher2 = st.selectbox("cHER2", ["0", "1+", "2+", "3+"])
her2_expr = st.selectbox("HER2 Expression", ["Positive", "Negative", "Unknown"])
us_size = st.number_input("Ultrasound Tumor Size (mm)", 0.0, 100.0)

# データフレーム構築
input_dict = {
    "age": age,
    "high(cm)": height,
    "Weight(kg)": weight,
    "Diagnositc": diagnostic,
    "menopause": menopause,
    "cT": ct,
    "CNB Histopathology": histology,
    "cHG": chg,
    "cER(%)": cer,
    "cPgR(%)": cpgr,
    "cHER2": cher2,
    "HER2 expression": her2_expr,
    "US size(mm)": us_size
}
input_df = pd.DataFrame([input_dict])

# カテゴリ変数をエンコード
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except:
            st.error(f"Invalid input for {col}: {input_df[col].values[0]}")
            st.stop()

# 予測
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    result = "Metastasis" if pred == 1 else "No Metastasis"
    st.success(f"Prediction: {result} ({prob * 100:.1f}%)")