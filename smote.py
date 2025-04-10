import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib


# Load dataset
df = pd.read_csv("cicp_dataset/train.csv")

# Prepare features and target
X = df.drop(columns=["policy_id", "is_claim"])
y = df["is_claim"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Fit and transform on full dataset
X_encoded = preprocessor.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE to training set
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Train the model
model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
model.fit(X_train_sm, y_train_sm)
joblib.dump(model, "rf_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
# --- Feature Importance ---
# Get feature names after encoding
encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([encoded_cat_features, X.drop(columns=categorical_cols).columns])

# Get importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Plot top 15 features
top_features = feature_importance_df.head(15)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Top 15 Most Important Features for Predicting Claims')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
