"""
FastAPI application for JIRA effort estimation prediction.
Provides a REST API endpoint to predict Actual_Time_Spent_hrs for JIRA issues.
"""

import os
import sys
import pickle
from typing import List
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Global variable to store loaded model data
model_data = None


def to_dict(pydantic_model):
    """
    Convert Pydantic model to dict, compatible with both v1 and v2.
    In Pydantic v1, use .dict(), in v2 use .model_dump().
    """
    if hasattr(pydantic_model, "model_dump"):
        return pydantic_model.model_dump()
    elif hasattr(pydantic_model, "dict"):
        return pydantic_model.dict()
    else:
        # Fallback: try to convert manually
        return dict(pydantic_model)


def get_script_directory():
    """
    Get the directory where the script is located.
    Handles cases where __file__ might not be defined.
    """
    if hasattr(sys, "frozen"):
        # PyInstaller or similar
        return os.path.dirname(sys.executable)
    elif "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    else:
        # Fallback to current working directory
        return os.getcwd()


def load_model():
    """
    Load the trained model and preprocessing components.
    Tries to load from jira_model.pkl first, then falls back to saved_models/ directory.
    Uses absolute paths based on the script's directory to avoid path issues.
    """
    global model_data

    # Get the directory where this script is located
    script_dir = get_script_directory()

    # Try loading from single file first
    jira_model_path = os.path.join(script_dir, "jira_model.pkl")
    if os.path.exists(jira_model_path):
        try:
            with open(jira_model_path, "rb") as f:
                model_data = pickle.load(f)
            # Validate that model_data has the expected structure
            if isinstance(model_data, dict) and all(
                key in model_data
                for key in ["model", "scaler", "label_encoders", "feature_columns"]
            ):
                print(f"[OK] Loaded model from {jira_model_path}")
                return model_data
            else:
                print(
                    "[WARN] jira_model.pkl exists but has unexpected structure, trying saved_models/"
                )
        except Exception as e:
            print(f"[WARN] Error loading jira_model.pkl: {e}, trying saved_models/")

    # Try loading from saved_models directory
    saved_models_dir = os.path.join(script_dir, "saved_models")
    best_model_path = os.path.join(saved_models_dir, "best_model.pkl")

    if os.path.exists(best_model_path):
        try:
            with open(best_model_path, "rb") as f:
                model = pickle.load(f)

            scaler_path = os.path.join(saved_models_dir, "scaler.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            label_encoders_path = os.path.join(saved_models_dir, "label_encoders.pkl")
            if not os.path.exists(label_encoders_path):
                raise FileNotFoundError(
                    f"Label encoders file not found: {label_encoders_path}"
                )
            with open(label_encoders_path, "rb") as f:
                label_encoders = pickle.load(f)

            feature_columns_path = os.path.join(saved_models_dir, "feature_columns.pkl")
            if not os.path.exists(feature_columns_path):
                raise FileNotFoundError(
                    f"Feature columns file not found: {feature_columns_path}"
                )
            with open(feature_columns_path, "rb") as f:
                feature_columns = pickle.load(f)

            model_data = {
                "model": model,
                "scaler": scaler,
                "label_encoders": label_encoders,
                "feature_columns": feature_columns,
            }
            print(f"[OK] Loaded model from {saved_models_dir}/")
            return model_data
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model files found in saved_models/ but error loading them: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Error loading model files from saved_models/: {e}")

    # If we get here, no model files were found
    raise FileNotFoundError(
        f"Model files not found. Checked:\n"
        f"  - {jira_model_path}\n"
        f"  - {saved_models_dir}/\n"
        f"Please ensure either 'jira_model.pkl' exists in the script directory "
        f"or 'saved_models/' directory contains the required model files."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    try:
        load_model()
        print("[OK] Model loaded successfully")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Script directory: {get_script_directory()}")
    except Exception as e:
        error_msg = f"[ERROR] Error loading model: {e}"
        print(error_msg)
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Failed to load model during startup: {e}") from e
    yield
    # Cleanup on shutdown (if needed)
    pass


app = FastAPI(
    title="JIRA Effort Estimation API",
    description="Predict Actual_Time_Spent_hrs for JIRA issues using machine learning",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models for request/response validation
class IssueInput(BaseModel):
    """Input model for JIRA issue features."""

    Issue_Type: str = Field(..., description="Type of issue (e.g., Bug, Story, Task)")
    Priority: str = Field(
        ..., description="Priority level (e.g., Low, Medium, High, Critical)"
    )
    Severity: str = Field(
        ..., description="Severity level (e.g., Trivial, Minor, Major, Blocker)"
    )
    Story_Points: float = Field(
        ..., ge=0, description="Story points assigned to the issue"
    )
    Epic_Size: float = Field(..., ge=0, description="Size of the epic")
    Number_of_Subtasks: int = Field(..., ge=0, description="Number of subtasks")
    Number_of_Linked_Issues: int = Field(
        ..., ge=0, description="Number of linked issues"
    )
    Number_of_Comments: int = Field(..., ge=0, description="Number of comments")
    Number_of_Attachments: int = Field(..., ge=0, description="Number of attachments")
    Number_of_Description_Edits: int = Field(
        ..., ge=0, description="Number of description edits"
    )
    Complexity: str = Field(
        ..., description="Complexity level (e.g., Low, Medium, High, Very High)"
    )
    Average_Historical_Hours: float = Field(
        ..., ge=0, description="Average historical hours for similar issues"
    )
    Assignee_Experience_yrs: float = Field(
        ..., ge=0, description="Assignee experience in years"
    )
    Team_Velocity: float = Field(..., ge=0, description="Team velocity")
    Team_Size: int = Field(..., ge=1, description="Team size")

    class Config:
        json_schema_extra = {
            "example": {
                "Issue_Type": "Story",
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
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction results."""

    predicted_hours: float = Field(..., description="Predicted Actual_Time_Spent_hrs")

    class Config:
        json_schema_extra = {"example": {"predicted_hours": 12.5}}


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions that includes all input fields plus prediction."""

    Issue_Type: str = Field(..., description="Type of issue (e.g., Bug, Story, Task)")
    Priority: str = Field(
        ..., description="Priority level (e.g., Low, Medium, High, Critical)"
    )
    Severity: str = Field(
        ..., description="Severity level (e.g., Trivial, Minor, Major, Blocker)"
    )
    Story_Points: float = Field(
        ..., ge=0, description="Story points assigned to the issue"
    )
    Epic_Size: float = Field(..., ge=0, description="Size of the epic")
    Number_of_Subtasks: int = Field(..., ge=0, description="Number of subtasks")
    Number_of_Linked_Issues: int = Field(
        ..., ge=0, description="Number of linked issues"
    )
    Number_of_Comments: int = Field(..., ge=0, description="Number of comments")
    Number_of_Attachments: int = Field(..., ge=0, description="Number of attachments")
    Number_of_Description_Edits: int = Field(
        ..., ge=0, description="Number of description edits"
    )
    Complexity: str = Field(
        ..., description="Complexity level (e.g., Low, Medium, High, Very High)"
    )
    Average_Historical_Hours: float = Field(
        ..., ge=0, description="Average historical hours for similar issues"
    )
    Assignee_Experience_yrs: float = Field(
        ..., ge=0, description="Assignee experience in years"
    )
    Team_Velocity: float = Field(..., ge=0, description="Team velocity")
    Team_Size: int = Field(..., ge=1, description="Team size")
    predicted_hours: float = Field(..., description="Predicted Actual_Time_Spent_hrs")

    class Config:
        json_schema_extra = {
            "example": {
                "Issue_Type": "Bug",
                "Priority": "Medium",
                "Severity": "Minor",
                "Story_Points": 8,
                "Epic_Size": 19,
                "Number_of_Subtasks": 4,
                "Number_of_Linked_Issues": 0,
                "Number_of_Comments": 6,
                "Number_of_Attachments": 1,
                "Number_of_Description_Edits": 4,
                "Complexity": "Medium",
                "Average_Historical_Hours": 6.4,
                "Assignee_Experience_yrs": 6.3,
                "Team_Velocity": 43,
                "Team_Size": 7,
                "predicted_hours": 17.9,
            }
        }


def predict_effort(issue_data: dict) -> float:
    """
    Predict Actual_Time_Spent_hrs for a new JIRA issue.

    Args:
        issue_data: Dictionary with issue features

    Returns:
        Predicted hours (float)
    """
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert to DataFrame
    df = pd.DataFrame([issue_data])

    # Ensure all required columns are present
    required_cols = model_data["feature_columns"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=422, detail=f"Missing required columns: {missing_cols}"
        )

    # Reorder columns to match training data
    df = df[required_cols]

    # Handle missing values
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

    # Ensure non-negative prediction
    return max(0.0, float(prediction))


def predict_effort_batch(issues_data: List[dict]) -> List[float]:
    """
    Predict Actual_Time_Spent_hrs for multiple JIRA issues efficiently.

    Args:
        issues_data: List of dictionaries with issue features

    Returns:
        List of predicted hours (floats)
    """
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not issues_data:
        raise HTTPException(status_code=422, detail="Empty list of issues provided")

    # Convert to DataFrame
    df = pd.DataFrame(issues_data)

    # Ensure all required columns are present
    required_cols = model_data["feature_columns"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=422, detail=f"Missing required columns: {missing_cols}"
        )

    # Reorder columns to match training data
    df = df[required_cols]

    # Handle missing values
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

    # Make predictions for all issues at once
    predictions = model_data["model"].predict(X_scaled)

    # Ensure non-negative predictions
    predictions = [max(0.0, float(pred)) for pred in predictions]

    return predictions


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "JIRA Effort Estimation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "predict_batch": "/predict/batch",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(issue: IssueInput):
    """
    Predict Actual_Time_Spent_hrs for a JIRA issue.

    Args:
        issue: IssueInput model with all required features

    Returns:
        PredictionResponse with predicted hours
    """
    try:
        predicted_hours = predict_effort(to_dict(issue))
        return PredictionResponse(predicted_hours=predicted_hours)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=List[BatchPredictionResponse])
async def predict_batch(request: Request):
    """
    Predict Actual_Time_Spent_hrs for multiple JIRA issues in batch.

    Accepts an array of issue objects and returns an array of predictions.
    This endpoint is optimized for processing multiple issues efficiently.

    Supports multiple input formats:
    - Direct array: [{"Issue_Type": "Bug", ...}, ...]
    - Wrapped object: {"issues": [{"Issue_Type": "Bug", ...}, ...]}
    - Single object: {"Issue_Type": "Bug", ...} (converted to single-item array)

    Returns:
        List of PredictionResponse objects with predicted hours
    """
    try:
        # Get raw JSON body
        try:
            body = await request.json()
        except Exception as json_error:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in request body: {str(json_error)}",
            )

        # Handle different input formats
        if isinstance(body, list):
            # Direct array format
            issues_list = body
        elif isinstance(body, dict):
            # Check if wrapped in an object
            if "issues" in body:
                issues_list = body["issues"]
            elif "data" in body:
                # Handle data field (could be array or single object)
                data_value = body["data"]
                if isinstance(data_value, list):
                    issues_list = data_value
                elif isinstance(data_value, dict):
                    issues_list = [data_value]
                else:
                    raise HTTPException(
                        status_code=422, detail="Data field must be an array or object"
                    )
            else:
                # Try to use the dict itself as a single-item list
                issues_list = [body]
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid input format. Expected array or object. Got: {type(body).__name__}",
            )

        if not issues_list:
            raise HTTPException(status_code=422, detail="Empty list of issues provided")

        # Validate and convert to IssueInput models
        issues_data = []
        validation_errors = []

        for idx, item in enumerate(issues_list):
            try:
                # Validate using Pydantic and convert to dict
                validated = IssueInput(**item)
                issues_data.append(to_dict(validated))
            except Exception as e:
                validation_errors.append(f"Issue {idx}: {str(e)}")

        if validation_errors:
            raise HTTPException(
                status_code=422,
                detail=f"Validation errors: {'; '.join(validation_errors[:5])}",  # Show first 5 errors
            )

        # Get batch predictions
        predicted_hours_list = predict_effort_batch(issues_data)

        # Create response with all input fields plus predictions
        predictions = []
        for issue_data, predicted_hours in zip(issues_data, predicted_hours_list):
            # Combine input data with prediction
            response_data = issue_data.copy()
            response_data["predicted_hours"] = predicted_hours
            predictions.append(BatchPredictionResponse(**response_data))

        return predictions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
