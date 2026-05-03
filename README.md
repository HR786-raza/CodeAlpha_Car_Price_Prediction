# 🚗 CodeAlpha-Car Price Prediction Dashboard

## 📌 Overview

This project is a complete **Data Science + Machine Learning
application** that predicts car prices using multiple ML models and
provides a modern interactive dashboard built with Streamlit.

------------------------------------------------------------------------

## 🎯 Features

-   📤 Upload your own dataset (CSV)
-   📊 Dataset preview and filtering
-   📈 Data visualization (scatter, histogram)
-   🔥 Correlation heatmap
-   🤖 Model comparison (Linear, RandomForest, DecisionTree)
-   ⭐ Feature importance visualization
-   💰 Car price prediction
-   📄 Download PDF report with charts

------------------------------------------------------------------------

## 🧠 Machine Learning Workflow

1.  Data Loading
2.  Data Preprocessing
    -   Handle missing values
    -   Feature engineering (Car Age)
    -   Encoding categorical variables
    -   Scaling numerical features
3.  Model Training
    -   Linear Regression
    -   Random Forest
    -   Decision Tree
4.  Model Evaluation (R2, RMSE)
5.  Best model selection
6.  Prediction

------------------------------------------------------------------------

## 📁 Project Structure

    car_price_prediction/
    │
    ├── data/
    ├── models/
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── train.py
    │   ├── predict.py
    │   ├── utils.py
    │   └── report.py
    │
    ├── app.py
    ├── main.py
    └── requirements.txt

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ How to Run

### 1. Train Models

``` bash
python main.py
```

### 2. Run Dashboard

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📊 Dataset Format

Your dataset should contain: - Car_Name - Year - Selling_Price
(target) - Present_Price - Driven_kms - Fuel_Type - Selling_type -
Transmission - Owner

------------------------------------------------------------------------

## 📈 Visualizations

-   Price vs Car Age
-   Price distribution
-   Correlation heatmap
-   Feature importance

------------------------------------------------------------------------

## 📄 PDF Report

Generated report includes: - Predicted price - Charts - Summary

------------------------------------------------------------------------

## 🚀 Future Improvements

-   SHAP explainability
-   Hyperparameter tuning
-   Deploy to cloud (Streamlit Cloud)

------------------------------------------------------------------------

## 👨‍💻 Author

Data Science Internship Project

------------------------------------------------------------------------

## ⭐ Notes

-   Ensure dataset format matches expected columns
-   Model performs best with realistic inputs
