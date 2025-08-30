🚗 Car Price Prediction with Gradient Boosting

This project predicts the selling price of used cars using Gradient Boosting Regressor.
The dataset is from CarDekho Vehicle Dataset




📌 Project Workflow

Define the Problem

Predict selling price of cars based on mileage, year, fuel type, transmission, etc.

Dataset

Source: Kaggle

Features:

year → Manufacturing year

km_driven → Distance driven

fuel → Type of fuel (Petrol, Diesel, etc.)

seller_type → Individual / Dealer

transmission → Manual / Automatic

owner → Ownership type

Target: selling_price

Exploratory Data Analysis (EDA)

Removed anomalies (e.g., extreme km_driven values).

Visualized distributions (histograms, scatter plots).

Checked correlations and feature impact.

Preprocessing

Applied StandardScaler for numeric features.

Applied OneHotEncoder for categorical features.

Built preprocessing pipeline using ColumnTransformer.

Model Training

Algorithm: Gradient Boosting Regressor

Used 5-fold Cross Validation.

Hyperparameter tuning via RandomizedSearchCV / GridSearchCV.

Evaluation Metrics

RMSE (Root Mean Squared Error)

R² Score



🖥️ Streamlit App

A simple Streamlit frontend allows users to enter car details and get predicted selling prices.

Run locally

streamlit run app.py


Inputs in app:

Car Age

Kilometers Driven

Fuel Type

Seller Type

Transmission

Ownership

Output:

Predicted Selling Price (in INR)



⚙️ Requirements

Main dependencies (pinned for Streamlit Cloud compatibility):

streamlit
pandas
numpy
scikit-learn==1.5.1
joblib==1.4.2
matplotlib
seaborn
scipy




📂 Project Structure

Boosting_task/
│
├── app.py                # Streamlit frontend
├── trained_model.py       # Training script
├── car_price_model.joblib # Saved model
├── Boosting.ipynb         # Notebook with EDA + training
├── requirements.txt       # Dependencies
└── README.md              # Project documentation



🚀 Future Improvements

Add more ML models (Random Forest, XGBoost) for comparison.

Deploy on Streamlit Cloud / Hugging Face Spaces.

Improve frontend UI with plots and insights. 



Screenshot for app:


<img width="909" height="818" alt="Screenshot 2025-08-30 080313" src="https://github.com/user-attachments/assets/20eb0648-e5b1-40a8-969c-7a14b4e29696" />
