"""
Flask-based Student Admission Prediction API
Simple alternative to BentoML for demonstration
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import hashlib
import jwt
import os
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from functools import wraps

app = Flask(__name__)

# Security configuration
SECRET_KEY = "demo-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Simple user database
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

# Load models
def load_models():
    """Load models from joblib files"""
    global model, scaler, feature_names
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        models_path = os.path.join(project_root, 'models')
        
        model = joblib.load(os.path.join(models_path, 'admission_model.joblib'))
        scaler = joblib.load(os.path.join(models_path, 'scaler.joblib'))
        feature_names = joblib.load(os.path.join(models_path, 'feature_names.joblib'))
        print("✅ Models loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False

# Initialize models
models_loaded = load_models()

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
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json()
        token = data.get("token") if data else None
        
        if not token:
            return jsonify({
                "error": "Authorization token required",
                "status": "error"
            }), 401
        
        user = verify_token(token)
        if not user:
            return jsonify({
                "error": "Invalid or expired token",
                "status": "error"
            }), 401
            
        return f(*args, **kwargs)
    return decorated_function

def validate_student_data(data: Dict[str, Any]) -> tuple:
    """Validate student input data"""
    required_fields = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate ranges
    try:
        if not (260 <= float(data['GRE_Score']) <= 340):
            return False, "GRE Score must be between 260 and 340"
        if not (80 <= float(data['TOEFL_Score']) <= 120):
            return False, "TOEFL Score must be between 80 and 120"
        if not (1 <= int(data['University_Rating']) <= 5):
            return False, "University Rating must be between 1 and 5"
        if not (1.0 <= float(data['SOP']) <= 5.0):
            return False, "SOP must be between 1.0 and 5.0"
        if not (1.0 <= float(data['LOR']) <= 5.0):
            return False, "LOR must be between 1.0 and 5.0"
        if not (6.0 <= float(data['CGPA']) <= 10.0):
            return False, "CGPA must be between 6.0 and 10.0"
        if int(data['Research']) not in [0, 1]:
            return False, "Research must be 0 or 1"
    except (ValueError, TypeError):
        return False, "Invalid data types in input"
    
    return True, ""

def interpret_prediction(chance: float) -> tuple:
    """Interpret the prediction result"""
    if chance >= 0.8:
        confidence = "High"
        interpretation = "Excellent admission chances! Your profile is very strong."
    elif chance >= 0.6:
        confidence = "Medium-High"
        interpretation = "Good admission chances. Consider strengthening weak areas."
    elif chance >= 0.4:
        confidence = "Medium"
        interpretation = "Moderate admission chances. Focus on improving your profile."
    elif chance >= 0.2:
        confidence = "Low"
        interpretation = "Lower admission chances. Significant improvement needed."
    else:
        confidence = "Very Low"
        interpretation = "Very low admission chances. Consider alternative options."
    
    return confidence, interpretation

def make_prediction(student_data: Dict[str, Any]) -> float:
    """Make a prediction using the loaded model"""
    if not models_loaded:
        raise Exception("Models not loaded")
    
    # Prepare features in correct order
    feature_order = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research']
    features = np.array([[float(student_data[field]) for field in feature_order]])
    
    # Make prediction
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    # Ensure prediction is within valid range
    return max(0.0, min(1.0, prediction))

# API Endpoints

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "admission_prediction_api",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "models_loaded": models_loaded
    })

@app.route('/login', methods=['POST'])
def login():
    """User authentication endpoint"""
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            return jsonify({
                "error": "Username and password required",
                "status": "error"
            }), 400
        
        user = authenticate_user(username, password)
        if not user:
            return jsonify({
                "error": "Invalid credentials",
                "status": "error"
            }), 401
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        
        return jsonify({
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "username": user["username"],
                "role": user["role"]
            },
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Login failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/predict', methods=['POST'])
@require_auth
def predict():
    """Single student admission prediction endpoint"""
    try:
        if not models_loaded:
            return jsonify({
                "error": "Models not loaded",
                "status": "error"
            }), 500
        
        data = request.get_json()
        
        # Extract student data
        student_data = data.get("student_data")
        if not student_data:
            return jsonify({
                "error": "Student data required",
                "status": "error"
            }), 400
        
        # Validate input data
        is_valid, error_msg = validate_student_data(student_data)
        if not is_valid:
            return jsonify({
                "error": f"Invalid input: {error_msg}",
                "status": "error"
            }), 400
        
        # Make prediction
        prediction = make_prediction(student_data)
        
        # Interpret prediction
        confidence_level, interpretation = interpret_prediction(prediction)
        
        return jsonify({
            "chance_of_admit": round(prediction, 4),
            "confidence_level": confidence_level,
            "interpretation": interpretation,
            "input_data": student_data,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/predict_batch', methods=['POST'])
@require_auth
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if not models_loaded:
            return jsonify({
                "error": "Models not loaded",
                "status": "error"
            }), 500
        
        data = request.get_json()
        
        # Extract students data
        students_data = data.get("students")
        if not students_data or not isinstance(students_data, list):
            return jsonify({
                "error": "Students data (list) required",
                "status": "error"
            }), 400
        
        predictions = []
        errors = []
        
        for i, student_data in enumerate(students_data):
            try:
                # Validate input data
                is_valid, error_msg = validate_student_data(student_data)
                if not is_valid:
                    errors.append(f"Student {i+1}: {error_msg}")
                    continue
                
                # Make prediction
                prediction = make_prediction(student_data)
                
                # Interpret prediction
                confidence_level, interpretation = interpret_prediction(prediction)
                
                predictions.append({
                    "student_id": i + 1,
                    "chance_of_admit": round(prediction, 4),
                    "confidence_level": confidence_level,
                    "interpretation": interpretation,
                    "input_data": student_data
                })
                
            except Exception as e:
                errors.append(f"Student {i+1}: {str(e)}")
        
        # Calculate summary statistics
        if predictions:
            admission_chances = [p["chance_of_admit"] for p in predictions]
            summary = {
                "total_students": len(students_data),
                "successful_predictions": len(predictions),
                "errors": len(errors),
                "average_chance": round(np.mean(admission_chances), 4),
                "min_chance": round(np.min(admission_chances), 4),
                "max_chance": round(np.max(admission_chances), 4),
                "std_chance": round(np.std(admission_chances), 4)
            }
        else:
            summary = {
                "total_students": len(students_data),
                "successful_predictions": 0,
                "errors": len(errors),
                "message": "No successful predictions"
            }
        
        return jsonify({
            "predictions": predictions,
            "summary": summary,
            "errors": errors if errors else None,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Batch prediction failed: {str(e)}",
            "status": "error"
        }), 500

if __name__ == "__main__":
    print("🚀 Student Admission Prediction API")
    print(f"Models loaded: {models_loaded}")
    print("Available endpoints: /health, /login, /predict, /predict_batch")
    app.run(host='0.0.0.0', port=3000, debug=True)
