"""
BentoML Service for Student Admission Prediction
Inspired by the accidents project structure
"""

import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.exceptions import BentoMLException
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException
import jwt
from datetime import datetime, timedelta

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "admission_prediction_secret_key_2025"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "admin": "admin123",
    "user": "user123", 
    "test": "test123",
    "demo": "demo123"
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    """JWT Authentication middleware for protected endpoints"""
    async def dispatch(self, request, call_next):
        # Only protect prediction endpoints
        protected_paths = [
            "/v1/models/admission_predictor/predict",
            "/v1/models/admission_predictor/predict_batch"
        ]
        
        if request.url.path in protected_paths:
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                # Remove 'Bearer ' prefix if present
                if token.startswith("Bearer "):
                    token = token[7:]
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                request.state.user = payload.get("sub")
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

        response = await call_next(request)
        return response

# Pydantic models for input validation
class StudentData(BaseModel):
    """Input model for student admission prediction"""
    GRE_Score: float = Field(..., ge=260, le=340, description="GRE Score (260-340)")
    TOEFL_Score: float = Field(..., ge=80, le=120, description="TOEFL Score (80-120)")
    University_Rating: int = Field(..., ge=1, le=5, description="University Rating (1-5)")
    SOP: float = Field(..., ge=1.0, le=5.0, description="Statement of Purpose Score (1.0-5.0)")
    LOR: float = Field(..., ge=1.0, le=5.0, description="Letter of Recommendation Score (1.0-5.0)")
    CGPA: float = Field(..., ge=6.0, le=10.0, description="CGPA (6.0-10.0)")
    Research: int = Field(..., ge=0, le=1, description="Research Experience (0 or 1)")

class BatchStudentData(BaseModel):
    """Input model for batch predictions"""
    students: list = Field(..., description="List of student data for batch prediction")

class LoginCredentials(BaseModel):
    """Login credentials model"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

# Load models from joblib files
import joblib
import os

MODEL_PATH = "/app/models"
try:
    # Load models from joblib files
    admission_model = joblib.load(os.path.join(MODEL_PATH, "admission_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_PATH, "feature_names.joblib"))
    print("✅ Models loaded from joblib files")
    MODELS_LOADED = True
except Exception as e:
    print(f"⚠️ Could not load models from joblib files: {e}")
    MODELS_LOADED = False
    admission_model = None
    scaler = None
    feature_names = None

# Create the BentoML service
service = bentoml.Service("admission_prediction_service", runners=[])

# Add JWT authentication middleware
service.add_asgi_middleware(JWTAuthMiddleware)

def create_jwt_token(user_id: str) -> str:
    """Create a JWT token for authentication"""
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration,
        "iat": datetime.utcnow()
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

def interpret_prediction(chance: float):
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

# API Endpoints

@service.api(input=bentoml.io.Text(), output=JSON())
def health(input_data: str = "") -> dict:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "admission_prediction_service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "models_loaded": MODELS_LOADED
    }

@service.api(input=JSON(pydantic_model=LoginCredentials), output=JSON())
async def login(credentials: LoginCredentials, ctx: bentoml.Context) -> dict:
    """Login endpoint for authentication"""
    username = credentials.username
    password = credentials.password
    
    # Check for empty credentials
    if not username or not password:
        ctx.response.status_code = 401
        return {"detail": "Invalid credentials"}

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 3600,
            "user": {
                "username": username
            },
            "status": "success"
        }
    else:
        # Set status code via context and return error message
        ctx.response.status_code = 401
        return {"detail": "Invalid credentials"}

@service.api(
    input=JSON(pydantic_model=StudentData),
    output=JSON(),
    route="v1/models/admission_predictor/predict"
)
async def predict(input_data: StudentData, ctx: bentoml.Context) -> dict:
    """Single student admission prediction endpoint"""
    if not MODELS_LOADED:
        raise ValueError("Models not available")
    
    try:
        # Get authenticated user
        request = ctx.request
        user = getattr(request.state, 'user', 'unknown')
        
        # Convert input data to numpy array in the correct feature order
        input_array = np.array([[
            input_data.GRE_Score,
            input_data.TOEFL_Score,
            input_data.University_Rating,
            input_data.SOP,
            input_data.LOR,
            input_data.CGPA,
            input_data.Research
        ]])
        
        # Scale the features
        scaled_features = scaler.transform(input_array)
        
        # Make prediction
        prediction = admission_model.predict(scaled_features)
        chance_of_admit = float(prediction[0])
        
        # Ensure prediction is within valid range
        chance_of_admit = max(0.0, min(1.0, chance_of_admit))
        
        # Interpret the prediction
        confidence_level, interpretation = interpret_prediction(chance_of_admit)
        
        return {
            "chance_of_admit": round(chance_of_admit, 4),
            "confidence_level": confidence_level,
            "interpretation": interpretation,
            "input_data": input_data.dict(),
            "user": user,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

@service.api(
    input=JSON(pydantic_model=BatchStudentData),
    output=JSON(),
    route="v1/models/admission_predictor/predict_batch"
)
async def predict_batch(input_data: BatchStudentData, ctx: bentoml.Context) -> dict:
    """Batch prediction endpoint for multiple students"""
    if not MODELS_LOADED:
        raise ValueError("Models not available")
    
    try:
        # Get authenticated user
        request = ctx.request
        user = getattr(request.state, 'user', 'unknown')
        
        predictions = []
        errors = []
        
        for i, student_dict in enumerate(input_data.students):
            try:
                # Create StudentData object from dict
                student = StudentData(**student_dict)
                
                # Convert to numpy array
                input_array = np.array([[
                    student.GRE_Score,
                    student.TOEFL_Score,
                    student.University_Rating,
                    student.SOP,
                    student.LOR,
                    student.CGPA,
                    student.Research
                ]])
                
                # Scale and predict
                # Scale the features and make prediction
                scaled_features = scaler.transform(input_array)
                prediction = admission_model.predict(scaled_features)
                chance_of_admit = float(prediction[0])
                
                # Ensure valid range
                chance_of_admit = max(0.0, min(1.0, chance_of_admit))
                
                # Interpret prediction
                confidence_level, interpretation = interpret_prediction(chance_of_admit)
                
                predictions.append({
                    "student_id": i + 1,
                    "chance_of_admit": round(chance_of_admit, 4),
                    "confidence_level": confidence_level,
                    "interpretation": interpretation,
                    "input_data": student.dict()
                })
                
            except Exception as e:
                errors.append(f"Student {i+1}: {str(e)}")
        
        # Calculate summary statistics
        if predictions:
            admission_chances = [p["chance_of_admit"] for p in predictions]
            summary = {
                "total_students": len(input_data.students),
                "successful_predictions": len(predictions),
                "errors": len(errors),
                "average_chance": round(np.mean(admission_chances), 4),
                "min_chance": round(np.min(admission_chances), 4),
                "max_chance": round(np.max(admission_chances), 4),
                "std_chance": round(np.std(admission_chances), 4)
            }
        else:
            summary = {
                "total_students": len(input_data.students),
                "successful_predictions": 0,
                "errors": len(errors),
                "message": "No successful predictions"
            }
        
        return {
            "predictions": predictions,
            "summary": summary,
            "errors": errors if errors else None,
            "user": user,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise ValueError(f"Batch prediction failed: {str(e)}")
