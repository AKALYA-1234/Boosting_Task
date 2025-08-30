# trained_model.py
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# 1. Load dataset
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# 2. Basic cleaning
df = df[df['km_driven'] >= 0]
km_cut = df['km_driven'].quantile(0.995)
df = df[df['km_driven'] <= km_cut]

# Feature Engineering
df['car_age'] = 2025 - df['year']   # calculate car age
X = df[['car_age', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
y = df['selling_price']
y_log = np.log1p(y)   # log-transform target

# 3. Preprocessing
numeric_features = ['car_age', 'km_driven']
numeric_transformer = StandardScaler()

categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. Model
gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

# 5. Pipeline
pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', gbr)
])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 7. Train
pipeline.fit(X_train, y_train)

# 8. Save model
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as car_price_model.pkl")
