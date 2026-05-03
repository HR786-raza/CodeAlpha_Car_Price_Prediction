import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.predict import load_model, predict
from src.utils import validate_input, format_price
from src.report import generate_report
from src.train import train_models, evaluate_models
from src.data_preprocessing import preprocess_data, split_data

st.set_page_config(layout="wide")

# ---------------- UPLOAD REQUIRED ----------------
st.sidebar.header("📤 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is None:
    st.warning("⚠️ Please upload a dataset to continue")
    st.stop()

# Load dataset ONLY if uploaded
df = pd.read_csv(file)

# Feature engineering
df["Car_Age"] = 2024 - df["Year"]

# ---------------- MODEL TRAINING ----------------
X, y, pre = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

models = train_models(X_train, y_train, pre)
results = evaluate_models(models, X_test, y_test)

model = load_model()

# ---------------- UI ----------------
st.title("🚗 Car Price Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Dataset", "📈 Models", "💰 Prediction"])

# ---------------- DATASET TAB ----------------
with tab1:
    st.subheader("Dataset Preview")

    st.write("Shape:", df.shape)
    st.dataframe(df, use_container_width=True)

    st.subheader("🔥 Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    st.pyplot(fig)

# ---------------- MODEL TAB ----------------
with tab2:
    st.subheader("Model Comparison")

    res_df = pd.DataFrame(results).T
    st.dataframe(res_df)

    fig, ax = plt.subplots()
    res_df["R2"].plot(kind="bar", ax=ax)
    ax.set_title("R2 Score Comparison")
    st.pyplot(fig)

    # Feature Importance
    try:
        imp = model.named_steps["regressor"].feature_importances_
        names = model.named_steps["preprocessor"].get_feature_names_out()

        fig, ax = plt.subplots()
        ax.barh(names, imp)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    except:
        st.info("Feature importance only available for tree models")

# ---------------- PREDICTION TAB ----------------
with tab3:
    st.subheader("Predict Price")

    col1, col2 = st.columns(2)

    with col1:
        present_price = st.number_input("Present Price", 0.0)
        driven_kms = st.number_input("Driven Kms", 0)
        owner = st.selectbox("Owner", [0, 1, 2])
        car_age = st.number_input("Car Age", 0)

    with col2:
        fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG"])
        sell = st.selectbox("Selling Type", ["Dealer", "Individual"])
        trans = st.selectbox("Transmission", ["Manual", "Automatic"])

    if st.button("Predict"):
        data = validate_input({
            "Present_Price": present_price,
            "Driven_kms": driven_kms,
            "Owner": owner,
            "Car_Age": car_age,
            "Fuel_Type": fuel,
            "Selling_type": sell,
            "Transmission": trans,
        })

        price = predict(model, data)

        st.success(f"💰 Price: {format_price(price)}")

        # Save chart
        chart_path = "chart.png"
        plt.figure()
        plt.hist(df["Selling_Price"])
        plt.savefig(chart_path)

        pdf = generate_report(format_price(price), [chart_path])

        with open(pdf, "rb") as f:
            st.download_button("📄 Download Report", f, file_name="report.pdf")