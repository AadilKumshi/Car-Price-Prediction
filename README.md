# 🚗 Car Price Prediction App

[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-brightgreen?style=for-the-badge&logo=streamlit)](https://car-price-prediction-aadil.streamlit.app/)

An interactive machine learning app that predicts the **selling price of a used car** based on key attributes such as brand, year, mileage, fuel type, and transmission.

Built using **Python**, **pandas**, **scikit-learn**, and **Streamlit** — this end-to-end project demonstrates real-world ML deployment and user-facing interface design.

---

## 📊 Features

- Predict used car prices instantly based on user input
- Intuitive interface with dropdowns, sliders, and buttons
- Linear Regression model trained on real-world automotive data
- Clean and minimal UI using Streamlit

---

## 📂 Dataset

- Source: `carproject.csv` (used car dataset)
- Features used:
  - `Brand`
  - `Body`
  - `Engine Type`
  - `Registration`
  - `Year`
  - `Mileage`

---

## 🧠 ML Model

- **Model Used:** Linear Regression
- **Libraries:** `scikit-learn`, `pandas`, `numpy`
- Preprocessing:
  - Dropped nulls
  - Removed irrelevant columns like `Model`
  - Applied one-hot encoding to categorical variables

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/AadilKumshi/car-price-prediction.git
cd car-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

