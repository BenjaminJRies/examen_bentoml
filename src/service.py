"""
BentoML Service for Student Admission Prediction
Inspired by the accidents project structure
"""

import numpy as np
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta
import hashlib

# Define input/output schemas
class StudentData(BaseModel):
    """Input schema for student admission prediction"""
    GRE_Score: float = Field(..., ge=260, le=340, description="GRE Score (260-340)")
    TOEFL_Score: float = Field(..., ge=80, le=120, description="TOEFL Score (80-120)")  
    University_Rating: int = Field(..., ge=1, le=5, description="University Rating (1-5)")
    SOP: float = Field(..., ge=1.0, le=5.0, description="Statement of Purpose Score (1.0-5.0)")
    LOR: float = Field(..., ge=1.0, le=5.0, description="Letter of Recommendation Score (1.0-5.0)")
    CGPA: float = Field(..., ge=6.0, le=10.0, description="CGPA (6.0-10.0)")
    Research: int = Field(..., ge=0, le=1, description="Research Experience (0 or 1)")

    @validator('GRE_Score')
    def validate_gre_score(cls, v):
        if not 260 <= v <= 340:
            raise ValueError('GRE Score must be between 260 and 340')
        return v

    @validator('TOEFL_Score')
    def validate_toefl_score(cls, v):
        if not 80 <= v <= 120:
            raise ValueError('TOEFL Score must be between 80 and 120')
        return v

class BatchStudentData(BaseModel):
    """Input schema for batch predictions"""
    students: List[StudentData] = Field(..., description="List of student data for batch prediction")

class LoginRequest(BaseModel):
    """Login request schema"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    chance_of_admit: float = Field(..., description="Predicted chance of admission (0-1)")
    confidence_level: str = Field(..., description="Confidence level of prediction")
    interpretation: str = Field(..., description="Human-readable interpretation")

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")

class LoginResponse(BaseModel):
    """Login response schema"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

# Load the latest models from BentoML model store
model_ref = bentoml.sklearn.get("admission_predictor:latest")
scaler_ref = bentoml.sklearn.get("admission_scaler:latest")

# Initialize the service
service = bentoml.Service("admission_prediction_api", runners=[
    model_ref.to_runner(),
    scaler_ref.to_runner()
])

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

def require_auth(func):
    """Decorator to require authentication"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract token from request headers
        auth_header = kwargs.get('ctx', {}).get('request', {}).get('headers', {}).get('authorization')
        if not auth_header:
            raise bentoml.exceptions.Unauthorized("Authorization header missing")
        
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != 'bearer':
                raise bentoml.exceptions.Unauthorized("Invalid authorization scheme")
        except ValueError:
            raise bentoml.exceptions.Unauthorized("Invalid authorization header format")
        
        user = verify_token(token)
        if not user:
            raise bentoml.exceptions.Unauthorized("Invalid or expired token")
        
        kwargs['current_user'] = user
        return func(*args, **kwargs)
    return wrapper

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

def interpret_prediction(prediction: float, student_data: StudentData) -> str:
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

@service.api(input=LoginRequest, output=LoginResponse)
def login(login_data: LoginRequest) -> LoginResponse:
    """
    Authenticate user and return JWT access token
    
    Test credentials:
    - Username: admin, Password: admin123
    - Username: user, Password: user123  
    - Username: test, Password: test123
    """
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise bentoml.exceptions.Unauthorized("Invalid username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@service.api(input=StudentData, output=PredictionResponse)
async def predict(student_data: StudentData, ctx: bentoml.Context) -> PredictionResponse:
    """
    Predict admission chance for a single student
    
    Requires: Bearer token in Authorization header
    
    Example request:
    {
        "GRE_Score": 320,
        "TOEFL_Score": 110,
        "University_Rating": 4,
        "SOP": 4.5,
        "LOR": 4.0,
        "CGPA": 8.5,
        "Research": 1
    }
    """
    # Authentication check
    auth_header = ctx.request.headers.get('authorization')
    if not auth_header:
        raise bentoml.exceptions.Unauthorized("Authorization header missing")
    
    try:
        scheme, token = auth_header.split()
        if scheme.lower() != 'bearer':
            raise bentoml.exceptions.Unauthorized("Invalid authorization scheme")
    except ValueError:
        raise bentoml.exceptions.Unauthorized("Invalid authorization header format")
    
    user = verify_token(token)
    if not user:
        raise bentoml.exceptions.Unauthorized("Invalid or expired token")
    
    # Prepare input data
    input_data = pd.DataFrame([{
        'GRE_Score': student_data.GRE_Score,
        'TOEFL_Score': student_data.TOEFL_Score,
        'University_Rating': student_data.University_Rating,
        'SOP': student_data.SOP,
        'LOR': student_data.LOR,
        'CGPA': student_data.CGPA,
        'Research': student_data.Research
    }])
    
    # Scale the input data
    input_scaled = await scaler_ref.to_runner().predict.async_run(input_data)
    
    # Make prediction
    prediction = await model_ref.to_runner().predict.async_run(input_scaled)
    chance_of_admit = float(prediction[0])
    
    # Ensure prediction is within valid range
    chance_of_admit = max(0.0, min(1.0, chance_of_admit))
    
    return PredictionResponse(
        chance_of_admit=chance_of_admit,
        confidence_level=get_confidence_level(chance_of_admit),
        interpretation=interpret_prediction(chance_of_admit, student_data)
    )

@service.api(input=BatchStudentData, output=BatchPredictionResponse)
async def predict_batch(batch_data: BatchStudentData, ctx: bentoml.Context) -> BatchPredictionResponse:
    """
    Predict admission chances for multiple students
    
    Requires: Bearer token in Authorization header
    
    Example request:
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
        ]
    }
    """
    # Authentication check
    auth_header = ctx.request.headers.get('authorization')
    if not auth_header:
        raise bentoml.exceptions.Unauthorized("Authorization header missing")
    
    try:
        scheme, token = auth_header.split()
        if scheme.lower() != 'bearer':
            raise bentoml.exceptions.Unauthorized("Invalid authorization scheme")
    except ValueError:
        raise bentoml.exceptions.Unauthorized("Invalid authorization header format")
    
    user = verify_token(token)
    if not user:
        raise bentoml.exceptions.Unauthorized("Invalid or expired token")
    
    # Prepare batch input data
    input_list = []
    for student in batch_data.students:
        input_list.append({
            'GRE_Score': student.GRE_Score,
            'TOEFL_Score': student.TOEFL_Score,
            'University_Rating': student.University_Rating,
            'SOP': student.SOP,
            'LOR': student.LOR,
            'CGPA': student.CGPA,
            'Research': student.Research
        })
    
    input_df = pd.DataFrame(input_list)
    
    # Scale the input data
    input_scaled = await scaler_ref.to_runner().predict.async_run(input_df)
    
    # Make predictions
    predictions = await model_ref.to_runner().predict.async_run(input_scaled)
    
    # Prepare response
    prediction_responses = []
    prediction_values = []
    
    for i, (student, pred) in enumerate(zip(batch_data.students, predictions)):
        chance_of_admit = float(pred)
        chance_of_admit = max(0.0, min(1.0, chance_of_admit))
        prediction_values.append(chance_of_admit)
        
        prediction_responses.append(PredictionResponse(
            chance_of_admit=chance_of_admit,
            confidence_level=get_confidence_level(chance_of_admit),
            interpretation=interpret_prediction(chance_of_admit, student)
        ))
    
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
    
    return BatchPredictionResponse(
        predictions=prediction_responses,
        summary=summary
    )

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
            "/health": "GET - Health check (no auth required)"
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

if __name__ == "__main__":
    # This allows running the service directly with: python service.py
    print("Starting BentoML Admission Prediction Service...")
    print("Available endpoints:")
    print("- POST /login - Authentication")
    print("- POST /predict - Single prediction")
    print("- POST /predict_batch - Batch prediction")
    print("- GET /health - Health check")
    print("\nTest credentials:")
    print("- Username: admin, Password: admin123")
    print("- Username: user, Password: user123")
    print("- Username: test, Password: test123")
