import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_model():
    data = pd.read_csv("Train_data.csv")
    data["maximum likelihood estimation meta 2"] = data["maximum likelihood estimation meta 2"].fillna(0)
    X = data.drop(columns=["maximum likelihood estimation meta 2"])
    y = data["maximum likelihood estimation meta 2"]
    for col in X.columns:
        if X[col].dtype == "object":
            X[col], _ = pd.factorize(X[col])  # カラム名維持
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X.fillna(0), y)
    return model, list(X.columns)

st.title("Lymph Node Metastasis Prediction")

# 入力欄
age = st.number_input("Age", 20, 90)
height = st.number_input("Height (cm)", 130, 200)
weight = st.number_input("Weight (kg)", 30, 120)
axillary = st.selectbox("Axillary Evaluation", ["Imaging", "FNA negative", "CNB negative"])
menopause = st.selectbox("Menopause", ["Pre", "Post"])
cT = st.selectbox("Clinical T stage", ["T1", "T2", "T3"])
histology = st.selectbox("CNB Histology", ["IDC", "ILC", "DCIS", "Other"])
cHG = st.selectbox("Histological Grade", ["1", "2", "3"])
er = st.slider("cER (%)", 0, 100)
pgr = st.slider("cPgR (%)", 0, 100)
her2 = st.selectbox("cHER2", ["0", "1+", "2+", "3+"])
her2_protein = st.selectbox("HER2 Protein", ["Positive", "Negative", "Unknown"])
us_size = st.number_input("US tumor size (mm)", 0.0, 100.0)

# 入力データ
input_data = pd.DataFrame([{
    "Age": age,
    "Height": height,
    "Weight": weight,
    "Axillary": ["Imaging", "FNA negative", "CNB negative"].index(axillary),
    "Menopause": ["Pre", "Post"].index(menopause),
    "cT": ["T1", "T2", "T3"].index(cT),
    "Histology": ["IDC", "ILC", "DCIS", "Other"].index(histology),
    "Grade": int(cHG),
    "ER": er,
    "PgR": pgr,
    "HER2": ["0", "1+", "2+", "3+"].index(her2),
    "HER2_Protein": ["Positive", "Negative", "Unknown"].index(her2_protein),
    "US_Size": us_size
}])

# モデルと特徴量名を取得
model, feature_names = load_model()

# 入力データをモデルのカラム順に再構成（すべての列を揃え、欠損は0埋め）
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[feature_names]

# 予測
if st.button("Predict"):
    proba = model.predict_proba(input_data)[0][1]  # 転移の確率（クラス1）
    prediction = model.predict(input_data)[0]
    result_label = 'Metastasis' if prediction == 1 else 'No Metastasis'
    st.success(f"Prediction result: {result_label} ({proba * 100:.1f}%)")