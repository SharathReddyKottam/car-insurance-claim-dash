# 🚗 Car Insurance Claim Predictor

A machine learning-powered Streamlit app that predicts the likelihood of a car insurance claim based on user inputs like car age, policy tenure, and region — complete with visual insights and risk breakdowns.

👉 **Live App**: [car-claim-predictor.streamlit.app](https://car-claim-predictor.streamlit.app)

---

## 🔍 Overview

This dashboard helps insurance analysts, underwriters, and business teams:
- Predict if a policyholder is likely to file a car insurance claim
- Visualize top factors contributing to claims
- Simulate outcomes by adjusting customer profiles

---

## 🎯 Key Features

✅ Clean, interactive Streamlit dashboard  
✅ Risk prediction with class imbalance handling (SMOTE)  
✅ Feature importance bar chart  
✅ Real-world input sliders and dropdowns  
✅ Friendly UX: human-readable area clusters & claim probability display

---

## 📊 Machine Learning Model

- Model: `RandomForestClassifier` with tuned depth & tree count for optimized size and performance
- Data balancing: SMOTE (Synthetic Minority Oversampling Technique)
- Target: `is_claim` (binary classification)

---

## 🚀 Live Demo

👉 Click here to try it out live:  
**[https://car-claim-predictor.streamlit.app](https://car-claim-predictor.streamlit.app)**

---

## 🛠️ Tech Stack

- Python (Pandas, Scikit-learn, Joblib)
- Streamlit (frontend/dashboard)
- Git + GitHub (version control & deployment)
- SMOTE (imbalanced-learn)

---

## 🧪 How to Run Locally

'''bash
git clone https://github.com/SharathReddyKottam/car-insurance-claim-dash.git
cd car-insurance-claim-dash
pip install -r requirements.txt
streamlit run app.py

---

## ✍️ Author

Built and maintained by **Sharath Reddy Kottam**  
📍 [LinkedIn](https://www.linkedin.com/in/sharath-kottam/) | 🧑‍💻 [GitHub](https://github.com/SharathReddyKottam)
