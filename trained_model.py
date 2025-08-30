import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# 1. Load dataset
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# 2. Clean data
df = df[df['km_driven'] >= 0]
km_cut = df['km_driven'].quantile(0.995)
df = df[df['km_driven'] <= km_cut]

# Add car age
df['car_age'] = 2025 - df['year']

# Features & target
X = df[['car_age', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
y = df['selling_price']
y_log = np.log1p(y)

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
gbr = GradientBoostingRegressor(random_state=42)
pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', gbr)
])

# 5. Hyperparameter tuning
param_dist = {
    'model__learning_rate': [0.01, 0.02, 0.05, 0.1],
    'model__n_estimators': [100, 150, 200, 300],
    'model__max_depth': [3, 4, 5],
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=12,
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("Training model...")
search.fit(X, y_log)

best_estimator = search.best_estimator_
print("✅ Best Params:", search.best_params_)

# 6. Save model
joblib.dump(best_estimator, "car_price_model.joblib")
print("🎉 Model saved as car_price_model.joblib")
