import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("jira_effort_estimation.csv")

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())

# Set target variable explicitly
target_col = "Actual_Time_Spent_hrs"

if target_col not in df.columns:
    raise ValueError(
        f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}"
    )

print(f"\nUsing '{target_col}' as target variable")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Handle missing values
X = X.fillna(X.mean(numeric_only=True))
y = y.fillna(y.mean())

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "predictions": y_pred,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }

    print(f"{name} Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

# Select best model based on R² score
best_model_name = max(results.keys(), key=lambda k: results[k]["R2"])
best_model = results[best_model_name]["model"]
print(f"\n{'=' * 50}")
print(f"Best Model: {best_model_name}")
print(f"{'=' * 50}")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Model comparison
metrics_df = pd.DataFrame(
    {
        "Model": list(results.keys()),
        "RMSE": [results[m]["RMSE"] for m in results.keys()],
        "R²": [results[m]["R2"] for m in results.keys()],
    }
)
axes[0].bar(metrics_df["Model"], metrics_df["R²"])
axes[0].set_title("Model Comparison (R² Score)")
axes[0].set_ylabel("R² Score")
axes[0].tick_params(axis="x", rotation=45)

# Plot 2: Actual vs Predicted for best model
axes[1].scatter(y_test, results[best_model_name]["predictions"], alpha=0.6)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[1].set_xlabel("Actual Values")
axes[1].set_ylabel("Predicted Values")
axes[1].set_title(f"Actual vs Predicted ({best_model_name})")

plt.tight_layout()
plt.show()

print("\nPipeline components saved:")
print("  - scaler: StandardScaler")
print(f"  - best_model: {best_model_name}")
print(f"  - label_encoders: {len(label_encoders)} encoders")
