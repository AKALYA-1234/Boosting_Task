import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset (update path if needed)
df = pd.read_csv(r"C:\Users\DELL\Downloads\CAR DETAILS FROM CAR DEKHO.csv",encoding='latin1')   # replace with actual dataset file
# Preprocess dataset
df['car_age'] = 2024 - df['year']   # convert year → age
df.drop("year", axis=1, inplace=True)

# Target and features
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Categorical and numeric columns
categorical_cols = ["fuel", "seller_type", "transmission", "owner", "name"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

# Pipeline with GradientBoosting
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model + features
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("car_price_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model training complete. Files saved: car_price_model.pkl, car_price_features.pkl")
