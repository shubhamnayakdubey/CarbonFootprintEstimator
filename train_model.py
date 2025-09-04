import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate synthetic dataset
np.random.seed(42)
N = 5000

electricity_usage = np.random.randint(100, 600, N)  # kWh/month
car_km = np.random.randint(0, 2000, N)              # km/month
flights_short = np.random.poisson(0.2, N)           # count/month
flights_long = np.random.poisson(0.05, N)           # count/month
diet_type = np.random.choice([0, 1, 2], N)          # 0=vegan,1=veg,2=non-veg
shopping_freq = np.random.randint(0, 20, N)         # orders/month
water_usage = np.random.randint(50, 500, N)         # liters/day

# emission factors
carbon_footprint = (
    electricity_usage * 0.42
    + car_km * 0.20
    + flights_short * 250
    + flights_long * 1100
    + np.select([diet_type==0, diet_type==1, diet_type==2], [50, 100, 200])
    + shopping_freq * 10
    + water_usage * 0.0003 * 30
)

df = pd.DataFrame({
    "electricity_usage": electricity_usage,
    "car_km": car_km,
    "flights_short": flights_short,
    "flights_long": flights_long,
    "diet_type": diet_type,
    "shopping_freq": shopping_freq,
    "water_usage": water_usage,
    "carbon_footprint": carbon_footprint
})

df.to_csv("carbon_footprint_dataset.csv", index=False)

# Step 2: Train models
X = df.drop("carbon_footprint", axis=1)
y = df["carbon_footprint"]

categorical = ["diet_type"]
numeric = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical)
], remainder="passthrough")

models = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
    "rf": RandomForestRegressor(n_estimators=100, random_state=42)
}

metrics = {}
best_rmse = float("inf")
best_model_name = None
best_pipeline = None

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in models.items():
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metrics[name] = {"rmse": rmse, "r2": r2}
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = name
        best_pipeline = pipeline

# Save model
joblib.dump(best_pipeline, "model_pipeline.joblib")

# Save metrics
with open("training_metrics.json", "w") as f:
    json.dump({"metrics": metrics, "best_model": best_model_name}, f, indent=2)

# If linear, save coefficients plot
if best_model_name == "linear":
    feature_names = preprocessor.get_feature_names_out()
    coefs = best_pipeline.named_steps["model"].coef_
    plt.figure(figsize=(10,6))
    plt.barh(feature_names, coefs)
    plt.title("Linear Regression Coefficients (kg CO2 per unit)")
    plt.tight_layout()
    plt.savefig("linear_coefficients.png")
    plt.close()

print("Training complete. Best model:", best_model_name)