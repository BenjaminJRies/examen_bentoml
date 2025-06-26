"""
BentoML Service for Student Admission Prediction API
This service provides secure endpoints for predicting university admission chances.
"""

import bentoml
import numpy as np
import pandas as pd
import hashlib
import jwt
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Load the latest models from BentoML model store
model_ref = bentoml.sklearn.get("admission_predictor:latest")
scaler_ref = bentoml.sklearn.get("admission_scaler:latest")

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Simple user database (in production, use a proper database)
USERS_DB = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin"
    },
    "user": {
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
        "role": "user"
    },
    "test": {
        "password_hash": hashlib.sha256("test123".encode()).hexdigest(),
        "role": "user"
    }
}

def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == password_hash

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials"""
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {"username": username, "role": user["role"]}

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return {"username": username}
    except jwt.PyJWTError:
        return None

def validate_student_data(data: Dict[str, Any]) -> bool:
    """Validate student input data"""
    required_fields = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate ranges
    if not (260 <= data['GRE_Score'] <= 340):
        return False
    if not (80 <= data['TOEFL_Score'] <= 120):
        return False
    if not (1 <= data['University_Rating'] <= 5):
        return False
    if not (1.0 <= data['SOP'] <= 5.0):
        return False
    if not (1.0 <= data['LOR'] <= 5.0):
        return False
    if not (6.0 <= data['CGPA'] <= 10.0):
        return False
    if data['Research'] not in [0, 1]:
        return False
    
    return True

def get_confidence_level(prediction: float) -> str:
    """Determine confidence level based on prediction value"""
    if prediction >= 0.8:
        return "High"
    elif prediction >= 0.6:
        return "Medium"
    elif prediction >= 0.4:
        return "Low"
    else:
        return "Very Low"

def interpret_prediction(prediction: float) -> str:
    """Provide human-readable interpretation of prediction"""
    percentage = prediction * 100
    
    if prediction >= 0.8:
        return f"Excellent admission chances ({percentage:.1f}%). Strong profile across all metrics."
    elif prediction >= 0.6:
        return f"Good admission chances ({percentage:.1f}%). Solid profile with room for improvement."
    elif prediction >= 0.4:
        return f"Moderate admission chances ({percentage:.1f}%). Consider strengthening weak areas."
    else:
        return f"Low admission chances ({percentage:.1f}%). Significant improvement needed in multiple areas."

# Initialize the service
service = bentoml.Service("admission_prediction_api", runners=[
    model_ref.to_runner(),
    scaler_ref.to_runner()
])

@service.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def login(login_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Authenticate user and return JWT access token
    
    Input JSON:
    {
        "username": "test",
        "password": "test123"
    }
    
    Test credentials:
    - Username: admin, Password: admin123
    - Username: user, Password: user123  
    - Username: test, Password: test123
    """
    username = login_data.get("username")
    password = login_data.get("password")
    
    if not username or not password:
        return {"error": "Username and password required", "status": 400}
    
    user = authenticate_user(username, password)
    if not user:
        return {"error": "Invalid username or password", "status": 401}
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "status": 200
    }

@service.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
async def predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict admission chance for a single student
    
    Input JSON:
    {
        "student_data": {
            "GRE_Score": 320,
            "TOEFL_Score": 110,
            "University_Rating": 4,
            "SOP": 4.5,
            "LOR": 4.0,
            "CGPA": 8.5,
            "Research": 1
        },
        "token": "your_jwt_token_here"
    }
    """
    # Extract token and student data
    token = input_data.get("token")
    student_data = input_data.get("student_data")
    
    if not token:
        return {"error": "Authentication token required", "status": 401}
    
    if not student_data:
        return {"error": "Student data required", "status": 400}
    
    # Verify token
    user = verify_token(token)
    if not user:
        return {"error": "Invalid or expired token", "status": 401}
    
    # Validate student data
    if not validate_student_data(student_data):
        return {"error": "Invalid student data format or values", "status": 400}
    
    try:
        # Prepare input data
        input_df = pd.DataFrame([{
            'GRE_Score': student_data['GRE_Score'],
            'TOEFL_Score': student_data['TOEFL_Score'],
            'University_Rating': student_data['University_Rating'],
            'SOP': student_data['SOP'],
            'LOR': student_data['LOR'],
            'CGPA': student_data['CGPA'],
            'Research': student_data['Research']
        }])
        
        # Scale the input data
        input_scaled = await scaler_ref.to_runner().predict.async_run(input_df)
        
        # Make prediction
        prediction = await model_ref.to_runner().predict.async_run(input_scaled)
        chance_of_admit = float(prediction[0])
        
        # Ensure prediction is within valid range
        chance_of_admit = max(0.0, min(1.0, chance_of_admit))
        
        return {
            "chance_of_admit": chance_of_admit,
            "confidence_level": get_confidence_level(chance_of_admit),
            "interpretation": interpret_prediction(chance_of_admit),
            "student_profile": student_data,
            "status": 200
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "status": 500}

@service.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
async def predict_batch(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict admission chances for multiple students
    
    Input JSON:
    {
        "students": [
            {
                "GRE_Score": 320,
                "TOEFL_Score": 110,
                "University_Rating": 4,
                "SOP": 4.5,
                "LOR": 4.0,
                "CGPA": 8.5,
                "Research": 1
            },
            {
                "GRE_Score": 310,
                "TOEFL_Score": 100,
                "University_Rating": 3,
                "SOP": 3.5,
                "LOR": 3.5,
                "CGPA": 7.8,
                "Research": 0
            }
        ],
        "token": "your_jwt_token_here"
    }
    """
    # Extract token and students data
    token = input_data.get("token")
    students = input_data.get("students", [])
    
    if not token:
        return {"error": "Authentication token required", "status": 401}
    
    if not students:
        return {"error": "Students data required", "status": 400}
    
    # Verify token
    user = verify_token(token)
    if not user:
        return {"error": "Invalid or expired token", "status": 401}
    
    # Validate all student data
    for i, student in enumerate(students):
        if not validate_student_data(student):
            return {"error": f"Invalid data for student {i+1}", "status": 400}
    
    try:
        # Prepare batch input data
        input_list = []
        for student in students:
            input_list.append({
                'GRE_Score': student['GRE_Score'],
                'TOEFL_Score': student['TOEFL_Score'],
                'University_Rating': student['University_Rating'],
                'SOP': student['SOP'],
                'LOR': student['LOR'],
                'CGPA': student['CGPA'],
                'Research': student['Research']
            })
        
        input_df = pd.DataFrame(input_list)
        
        # Scale the input data
        input_scaled = await scaler_ref.to_runner().predict.async_run(input_df)
        
        # Make predictions
        predictions = await model_ref.to_runner().predict.async_run(input_scaled)
        
        # Prepare response
        prediction_results = []
        prediction_values = []
        
        for i, (student, pred) in enumerate(zip(students, predictions)):
            chance_of_admit = float(pred)
            chance_of_admit = max(0.0, min(1.0, chance_of_admit))
            prediction_values.append(chance_of_admit)
            
            prediction_results.append({
                "student_id": i + 1,
                "chance_of_admit": chance_of_admit,
                "confidence_level": get_confidence_level(chance_of_admit),
                "interpretation": interpret_prediction(chance_of_admit),
                "student_profile": student
            })
        
        # Calculate summary statistics
        summary = {
            "total_students": len(prediction_values),
            "average_admission_chance": float(np.mean(prediction_values)),
            "highest_chance": float(np.max(prediction_values)),
            "lowest_chance": float(np.min(prediction_values)),
            "high_confidence_count": len([p for p in prediction_values if p >= 0.8]),
            "medium_confidence_count": len([p for p in prediction_values if 0.6 <= p < 0.8]),
            "low_confidence_count": len([p for p in prediction_values if p < 0.6])
        }
        
        return {
            "predictions": prediction_results,
            "summary": summary,
            "status": 200
        }
        
    except Exception as e:
        return {"error": f"Batch prediction failed: {str(e)}", "status": 500}

@service.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def health() -> Dict[str, Any]:
    """
    Health check endpoint (no authentication required)
    """
    return {
        "status": "healthy",
        "service": "admission_prediction_api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "model_info": {
            "model_name": "admission_predictor",
            "scaler_name": "admission_scaler",
            "features": [
                "GRE_Score", "TOEFL_Score", "University_Rating", 
                "SOP", "LOR", "CGPA", "Research"
            ]
        },
        "endpoints": {
            "/login": "POST - Authenticate and get access token",
            "/predict": "POST - Single prediction (requires auth)",
            "/predict_batch": "POST - Batch prediction (requires auth)",
            "/health": "POST - Health check (no auth required)"
        },
        "authentication": {
            "type": "JWT Bearer Token",
            "test_credentials": {
                "admin": "admin123",
                "user": "user123",
                "test": "test123"
            }
        }
    }
