"""
Test script for JIRA effort estimation prediction model.
This script demonstrates how to use the trained model to predict Actual_Time_Spent_hrs.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os


def train_and_save_model():
    """
    Train the model and save all necessary components for later use.
    This function replicates the training logic from jira_prediction.py
    and saves the model components.
    """
    # Load dataset
    df = pd.read_csv("jira_effort_estimation.csv")

    # Set target variable
    target_col = "Actual_Time_Spent_hrs"

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}"
        )

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
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }

        print(f"{name} Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]["R2"])
    best_model = results[best_model_name]["model"]

    print(f"\n{'=' * 50}")
    print(f"Best Model: {best_model_name}")
    print(f"{'=' * 50}")

    # Save model components
    model_data = {
        "model": best_model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_columns": list(X.columns),
        "model_name": best_model_name,
        "metrics": results[best_model_name],
    }

    with open("jira_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("\nModel saved to 'jira_model.pkl'")
    return model_data, X_test, y_test


def load_model():
    """Load the trained model and preprocessing components."""
    if not os.path.exists("jira_model.pkl"):
        raise FileNotFoundError(
            "Model file 'jira_model.pkl' not found. Please run train_and_save_model() first."
        )

    with open("jira_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    return model_data


def predict_effort(model_data, issue_data):
    """
    Predict Actual_Time_Spent_hrs for a new JIRA issue.

    Args:
        model_data: Dictionary containing model, scaler, label_encoders, etc.
        issue_data: Dictionary or DataFrame with issue features

    Returns:
        Predicted hours (float)
    """
    # Convert to DataFrame if it's a dict
    if isinstance(issue_data, dict):
        df = pd.DataFrame([issue_data])
    else:
        df = issue_data.copy()

    # Ensure all required columns are present
    required_cols = model_data["feature_columns"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Reorder columns to match training data
    df = df[required_cols]

    # Handle missing values (use mean from training - simplified here)
    # In production, you'd want to save the mean values during training
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Encode categorical variables
    label_encoders = model_data["label_encoders"]
    for col in df.select_dtypes(include=["object"]).columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            df[col] = df[col].astype(str)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # Scale features
    X_scaled = model_data["scaler"].transform(df)

    # Make prediction
    prediction = model_data["model"].predict(X_scaled)[0]

    return max(0, prediction)  # Ensure non-negative prediction


def test_examples():
    """Test the model with example JIRA issues."""
    print("=" * 60)
    print("JIRA Effort Estimation - Test Examples")
    print("=" * 60)

    # Load or train model
    try:
        model_data = load_model()
        print("\n✓ Loaded existing model")
    except FileNotFoundError:
        print("\n⚠ Model not found. Training new model...")
        model_data, _, _ = train_and_save_model()

    print(f"\nUsing model: {model_data['model_name']}")
    print(f"Model R² Score: {model_data['metrics']['R2']:.4f}")
    print(f"Model RMSE: {model_data['metrics']['RMSE']:.4f}")

    # Example test cases
    test_cases = [
        {
            "name": "Simple Bug Fix",
            "data": {
                "Issue_Type": "Bug",
                "Priority": "Medium",
                "Severity": "Minor",
                "Story_Points": 3,
                "Epic_Size": 20,
                "Number_of_Subtasks": 1,
                "Number_of_Linked_Issues": 0,
                "Number_of_Comments": 3,
                "Number_of_Attachments": 0,
                "Number_of_Description_Edits": 1,
                "Complexity": "Low",
                "Average_Historical_Hours": 5.0,
                "Assignee_Experience_yrs": 3.0,
                "Team_Velocity": 45,
                "Team_Size": 8,
            },
        },
        {
            "name": "Complex Feature Story",
            "data": {
                "Issue_Type": "Story",
                "Priority": "High",
                "Severity": "Major",
                "Story_Points": 13,
                "Epic_Size": 100,
                "Number_of_Subtasks": 5,
                "Number_of_Linked_Issues": 3,
                "Number_of_Comments": 15,
                "Number_of_Attachments": 2,
                "Number_of_Description_Edits": 5,
                "Complexity": "Very High",
                "Average_Historical_Hours": 25.0,
                "Assignee_Experience_yrs": 8.0,
                "Team_Velocity": 50,
                "Team_Size": 10,
            },
        },
        {
            "name": "Medium Task",
            "data": {
                "Issue_Type": "Task",
                "Priority": "Medium",
                "Severity": "Minor",
                "Story_Points": 5,
                "Epic_Size": 50,
                "Number_of_Subtasks": 2,
                "Number_of_Linked_Issues": 1,
                "Number_of_Comments": 8,
                "Number_of_Attachments": 1,
                "Number_of_Description_Edits": 2,
                "Complexity": "Medium",
                "Average_Historical_Hours": 12.0,
                "Assignee_Experience_yrs": 5.0,
                "Team_Velocity": 45,
                "Team_Size": 9,
            },
        },
        {
            "name": "Critical Bug",
            "data": {
                "Issue_Type": "Bug",
                "Priority": "Critical",
                "Severity": "Blocker",
                "Story_Points": 8,
                "Epic_Size": 30,
                "Number_of_Subtasks": 3,
                "Number_of_Linked_Issues": 2,
                "Number_of_Comments": 12,
                "Number_of_Attachments": 1,
                "Number_of_Description_Edits": 4,
                "Complexity": "High",
                "Average_Historical_Hours": 15.0,
                "Assignee_Experience_yrs": 7.0,
                "Team_Velocity": 40,
                "Team_Size": 7,
            },
        },
    ]

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        try:
            predicted_hours = predict_effort(model_data, test_case["data"])
            print(f"\n{i}. {test_case['name']}")
            print(f"   Predicted Time: {predicted_hours:.2f} hours")
            print(f"   Story Points: {test_case['data']['Story_Points']}")
            print(f"   Complexity: {test_case['data']['Complexity']}")
            print(f"   Issue Type: {test_case['data']['Issue_Type']}")
        except Exception as e:
            print(f"\n{i}. {test_case['name']}")
            print(f"   ❌ Error: {str(e)}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


def test_batch_predictions():
    """Test batch predictions on multiple issues."""
    print("\n" + "=" * 60)
    print("BATCH PREDICTION TEST")
    print("=" * 60)

    try:
        model_data = load_model()
    except FileNotFoundError:
        print("⚠ Model not found. Training new model...")
        model_data, _, _ = train_and_save_model()

    # Create batch of test issues
    batch_data = pd.DataFrame(
        [
            {
                "Issue_Type": "Story",
                "Priority": "Medium",
                "Severity": "Minor",
                "Story_Points": 5,
                "Epic_Size": 50,
                "Number_of_Subtasks": 2,
                "Number_of_Linked_Issues": 1,
                "Number_of_Comments": 6,
                "Number_of_Attachments": 0,
                "Number_of_Description_Edits": 2,
                "Complexity": "Medium",
                "Average_Historical_Hours": 10.0,
                "Assignee_Experience_yrs": 4.0,
                "Team_Velocity": 45,
                "Team_Size": 8,
            },
            {
                "Issue_Type": "Task",
                "Priority": "Low",
                "Severity": "Trivial",
                "Story_Points": 2,
                "Epic_Size": 25,
                "Number_of_Subtasks": 0,
                "Number_of_Linked_Issues": 0,
                "Number_of_Comments": 2,
                "Number_of_Attachments": 0,
                "Number_of_Description_Edits": 1,
                "Complexity": "Low",
                "Average_Historical_Hours": 3.0,
                "Assignee_Experience_yrs": 2.0,
                "Team_Velocity": 40,
                "Team_Size": 6,
            },
        ]
    )

    predictions = []
    for idx, row in batch_data.iterrows():
        pred = predict_effort(model_data, row.to_dict())
        predictions.append(pred)

    batch_data["Predicted_Hours"] = predictions

    print("\nBatch Prediction Results:")
    print(
        batch_data[
            ["Issue_Type", "Story_Points", "Complexity", "Predicted_Hours"]
        ].to_string(index=False)
    )
    print(f"\nTotal Predicted Hours: {sum(predictions):.2f}")


if __name__ == "__main__":
    # Run test examples
    test_examples()

    # Run batch prediction test
    test_batch_predictions()
