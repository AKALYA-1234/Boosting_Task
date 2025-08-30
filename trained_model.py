import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# 1. Load dataset
data = pd.read_csv(r"C:\Users\DELL\Downloads\CAR DETAILS FROM CAR DEKHO.csv",encoding='latin1')

# 2. Drop 'name' (not useful for training)
data = data.drop(columns=["name"])

# 3. Create car_age column
current_year = 2025   # or you can use data["year"].max() if you want relative
data["car_age"] = current_year - data["year"]

# 4. Define features (X) and target (y)
X = data.drop(columns=["selling_price"])
y = data["selling_price"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Preprocess (categorical encoding + numeric passthrough)
categorical_cols = ["fuel", "seller_type", "transmission", "owner"]
numeric_cols = ["year", "km_driven", "car_age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# 7. Build pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor())
])

# 8. Train the model
pipeline.fit(X_train, y_train)

# 9. Save pipeline
joblib.dump(pipeline, "car_price_model.pkl")

print("✅ Model trained and saved as car_price_model.pkl")
