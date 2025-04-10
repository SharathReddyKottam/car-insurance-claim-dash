import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load model and preprocessor ---
model = joblib.load("rf_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# --- Load dataset for reference ---
df = pd.read_csv("cicp_dataset/train.csv")
X = df.drop(columns=["policy_id", "is_claim"])
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# --- Area Cluster Mapping (User-Friendly) ---
fake_cluster_mapping = {
    "C1": "Metro Central", "C2": "Urban North", "C3": "Urban East", "C4": "Urban South",
    "C5": "Urban West", "C6": "Suburban North", "C7": "Suburban East", "C8": "Suburban South",
    "C9": "Suburban West", "C10": "Rural North", "C11": "Rural East", "C12": "Rural South",
    "C13": "Rural West", "C14": "Industrial Belt", "C15": "Highway Zone"
}
reverse_cluster_mapping = {v: k for k, v in fake_cluster_mapping.items()}

# --- App Title ---
st.set_page_config(page_title="Insurance Claim Predictor", layout="wide")
st.title("🚗 Car Insurance Claim Prediction")
st.markdown("This dashboard uses machine learning to predict the likelihood of a policyholder filing a car insurance claim.")

st.markdown("---")

# --- Layout: Feature Importance + Form ---
col1, col2 = st.columns([1.3, 1])

# Left Column: Feature Importance
with col1:
    st.subheader("📊 Top Predictive Features")
    encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = np.concatenate([encoded_cat_features, X.drop(columns=categorical_cols).columns])
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x="importance", y="feature", data=importance_df, ax=ax)
    st.pyplot(fig)

# Right Column: Prediction Form
with col2:
    st.subheader("📝 Customer Profile")

    # Organize input fields
    c1, c2 = st.columns(2)
    car_age = c1.slider("Age of Car (yrs)", 0, 20, 5)
    policy_tenure = c2.slider("Policy Tenure (yrs)", 0, 15, 3)

    c3, c4 = st.columns(2)
    holder_age = c3.slider("Policyholder Age", 18, 80, 35)
    population_density = c4.number_input("Population Density", min_value=0, max_value=100000, value=5000)

    fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].dropna().unique()))
    cluster_label = st.selectbox("Area Cluster", list(fake_cluster_mapping.values()))
    area_cluster = reverse_cluster_mapping[cluster_label]

    st.markdown("")

    # Predict Button
    if st.button("🔍 Predict Claim Risk"):
        input_data = pd.DataFrame([{
            "age_of_car": car_age / 20,
            "policy_tenure": policy_tenure / 15,
            "age_of_policyholder": holder_age / 80,
            "population_density": population_density,
            "fuel_type": fuel_type,
            "area_cluster": area_cluster
        }])

        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = df[col].mode()[0]

        # Predict
        X_transformed = preprocessor.transform(input_data)
        prediction = model.predict(X_transformed)[0]
        proba = model.predict_proba(X_transformed)[0][1]

        st.markdown("---")
        st.subheader("📈 Prediction Outcome")

        if prediction == 1:
            st.error(f"🚨 High Risk of Claim")
            st.metric("Claim Probability", f"{proba:.1%}")
        else:
            st.success(f"✅ Low Risk of Claim")
            st.metric("Claim Probability", f"{proba:.1%}")

        st.progress(proba)
