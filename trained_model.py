# trained_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\CAR DETAILS FROM CAR DEKHO.csv",encoding='latin1')

# Drop rows with missing target values
df = df.dropna(subset=["selling_price"])

# Define features and target
X = df.drop(columns=["selling_price"])
y = df["selling_price"]

# Encode categorical features
categorical_cols = ["name", "fuel", "seller_type", "transmission", "owner"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "car_price_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("✅ Model and encoders saved successfully!")

