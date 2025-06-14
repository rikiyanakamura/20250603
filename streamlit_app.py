import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def train_model():
    df = pd.read_csv("Train_data.csv")
    features = [
        "age", "high(cm)", "Weight(kg)", "Diagnositc", "menopause",
        "cT", "CNB Histopathology", "cHG", "cER(%)", "cPgR(%)",
        "cHER2", "HER2 expression", "US size(mm)"
    ]
    target = "Ln meta."

    df = df[features + [target]].dropna()
    df[target] = df[target].map({"Negative": 0, "Positive": 1})

    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    X = df[features]
    y = df[target]

    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X, y)
    return model, features, label_encoders

model, features, label_encoders = train_model()

st.title("Lymph Node Metastasis Prediction (Safe Encoding Version)")

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

# データ構築
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

# 安全なカテゴリ変換：未知の値は -1 に
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# 予測
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    result = "Metastasis" if pred == 1 else "No Metastasis"
    st.success(f"Prediction: {result} ({prob * 100:.1f}%)")