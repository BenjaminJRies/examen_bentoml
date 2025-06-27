# BentoML Student Admission Prediction API - Exam Project

## 📋 Project Overview

This project implements a complete Student Admission Prediction system using BentoML with:
- **Machine Learning Pipeline**: Linear Regression model for predicting university admission chances
- **Secure REST API**: JWT-authenticated BentoML service with multiple endpoints
- **Docker Containerization**: Fully containerized application ready for deployment
- **Comprehensive Testing**: Complete unit test suite with 100% pass rate

## 🎯 Model Performance
- **Algorithm**: Linear Regression with StandardScaler
- **Performance**: R² Score = 0.8188 (81.88% accuracy)
- **Features**: 7 input features (GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research)

## 🚀 Quick Start Guide

### Prerequisites
- Docker installed and running
- Python 3.8+ (if running tests locally)
- Git (for cloning)

### Step 1: Extract and Setup
```bash
# Extract the Docker image (if provided as tar file)
docker load -i benjaminries_admission_prediction.tar

# Verify the image is loaded
docker images | grep benjaminries_admission_prediction
```

### Step 2: Run the Containerized API
```bash
# Start the container
docker run -d --name admission_api -p 3000:3000 benjaminries_admission_prediction:latest

# Check container status
docker ps

# View container logs (if needed)
docker logs admission_api
```

### Step 3: Verify API is Running
```bash
# Test health endpoint
curl -X POST http://localhost:3000/health -d ""

# Expected response:
# {"status":"healthy","service":"admission_prediction_service","timestamp":"...","version":"1.0.0","models_loaded":true}
```

## 🔐 API Authentication & Usage

### 1. Login to Get JWT Token
```bash
curl -X POST http://localhost:3000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Response will include access_token - save this for subsequent requests
```

### 2. Make Predictions
```bash
# Single prediction (replace YOUR_JWT_TOKEN with actual token)
curl -X POST http://localhost:3000/v1/models/admission_predictor/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "GRE_Score": 320,
    "TOEFL_Score": 110,
    "University_Rating": 3,
    "SOP": 4.0,
    "LOR": 4.5,
    "CGPA": 8.5,
    "Research": 1
  }'
```

### 3. Batch Predictions
```bash
curl -X POST http://localhost:3000/v1/models/admission_predictor/predict_batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "students": [
      {
        "GRE_Score": 320,
        "TOEFL_Score": 110,
        "University_Rating": 3,
        "SOP": 4.0,
        "LOR": 4.5,
        "CGPA": 8.5,
        "Research": 1
      },
      {
        "GRE_Score": 300,
        "TOEFL_Score": 100,
        "University_Rating": 2,
        "SOP": 3.0,
        "LOR": 3.5,
        "CGPA": 7.5,
        "Research": 0
      }
    ]
  }'
```

## 🧪 Running Unit Tests

### Method 1: Run Tests Against Running Container
```bash
# Ensure the container is running first
docker ps | grep admission_api

# Install test dependencies (if not already installed)
pip install requests pytest pyjwt

# Run the unit tests
pytest test_unit_tests.py -v

# Expected output: 16 tests, all PASSED
```

### Method 2: Run Tests from Source (Alternative)
```bash
# Clone the repository (if working from source)
git clone <repository_url>
cd examen_bentoml

# Setup virtual environment
python -m venv bentoml_env
source bentoml_env/bin/activate  # On Windows: bentoml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the service locally (in background)
python -m bentoml serve src.service_new:service --port 3000 &

# Run tests
pytest test_unit_tests.py -v
```

## 📋 API Endpoints

| Endpoint | Method | Authentication | Description |
|----------|--------|----------------|-------------|
| `/health` | POST | None | Health check endpoint |
| `/login` | POST | None | Authenticate and get JWT token |
| `/v1/models/admission_predictor/predict` | POST | JWT Required | Single student prediction |
| `/v1/models/admission_predictor/predict_batch` | POST | JWT Required | Batch student predictions |

## 📊 Input Data Format

### Student Data Fields
- **GRE_Score**: Float (260-340) - GRE test score
- **TOEFL_Score**: Float (80-120) - TOEFL test score  
- **University_Rating**: Integer (1-5) - University rating
- **SOP**: Float (1.0-5.0) - Statement of Purpose score
- **LOR**: Float (1.0-5.0) - Letter of Recommendation score
- **CGPA**: Float (6.0-10.0) - Cumulative Grade Point Average
- **Research**: Integer (0 or 1) - Research experience (0=No, 1=Yes)

### Authentication
- **Username**: "admin"
- **Password**: "admin123"

## 🔧 Troubleshooting

### Container Issues
```bash
# Check container status
docker ps -a

# View container logs
docker logs admission_api

# Restart container
docker restart admission_api

# Remove and recreate container
docker stop admission_api
docker rm admission_api
docker run -d --name admission_api -p 3000:3000 benjaminries_admission_prediction:latest
```

### Port Conflicts
```bash
# If port 3000 is busy, use different port
docker run -d --name admission_api -p 8080:3000 benjaminries_admission_prediction:latest

# Then access API at http://localhost:8080
```

### Test Failures
```bash
# Ensure container is running and healthy
curl -X POST http://localhost:3000/health -d ""

# Check if all dependencies are installed
pip list | grep -E "(requests|pytest|pyjwt)"

# Run tests with more verbose output
pytest test_unit_tests.py -v -s
```

## 📁 Project Structure

```
examen_bentoml/
├── src/
│   ├── service_new.py          # Main BentoML service
│   ├── prepare_data.py         # Data preparation script
│   └── train_model.py          # Model training script
├── data/
│   ├── raw/
│   │   └── admission.csv       # Original dataset
│   └── processed/              # Train/test splits
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/                     # Saved model files
├── test_unit_tests.py          # Unit test suite
├── Dockerfile                  # Container configuration
├── bentofile.yaml             # BentoML configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ✅ Unit Test Coverage

The test suite (`test_unit_tests.py`) includes 16 comprehensive tests:

### JWT Authentication Tests (4 tests)
- Valid JWT token validation
- Invalid JWT token handling
- Missing JWT token handling  
- Expired JWT token handling

### Login API Tests (4 tests)
- Valid credentials return JWT token
- Invalid credentials return 401
- Empty credentials return 401
- Missing credentials return error

### Prediction API Tests (6 tests)
- Valid prediction returns result
- Prediction without JWT returns 401
- Prediction with invalid JWT returns 401
- Invalid input data returns error
- Missing required fields return error
- Batch prediction functionality

### Health Endpoint Tests (1 test)
- Health endpoint accessibility

### Integration Tests (1 test)
- Complete workflow (login → predict)

**All tests must return PASSED status for successful evaluation.**

## 📦 Deliverables Summary

1. **README.md**: This comprehensive guide
2. **benjaminries_admission_prediction.tar**: Docker image archive (1.8GB)
3. **test_unit_tests.py**: Complete unit test suite (16 tests, all passing)

## 🎉 Success Criteria

✅ **API Functionality**: All endpoints work correctly  
✅ **Authentication**: JWT-based security implemented  
✅ **Model Performance**: 81.88% R² score achieved  
✅ **Containerization**: Docker image builds and runs successfully  
✅ **Testing**: 16/16 unit tests pass  
✅ **Documentation**: Complete setup and usage instructions  

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure Docker is running and has sufficient resources
4. Check container logs for detailed error messages

---

**Project completed successfully for BentoML Exam - June 27, 2025**

Afin de pouvoir commencer le projet vous devez suivre les étapes suivantes:

- Forker le projet sur votre compte github
- Cloner le projet sur votre machine
- Créer un environnement virtuel:
  ```bash
  python3 -m venv bentoml_env
  source bentoml_env/bin/activate  # Linux/Mac
  # ou bentoml_env\Scripts\activate  # Windows
  ```
- Installer les dépendances:
  ```bash
  pip install -r requirements.txt
  ```
- Récuperer le jeu de données à partir du lien suivant: [Lien de téléchargement](https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv)

## Dataset

The dataset contains information about student admissions to universities with the following variables:

- **GRE Score**: Score obtained on the GRE test (scored out of 340)
- **TOEFL Score**: Score obtained on the TOEFL test (scored out of 120)
- **University Rating**: University rating (scored out of 5)
- **SOP**: Statement of Purpose (scored out of 5)
- **LOR**: Letter of Recommendation (scored out of 5)
- **CGPA**: Cumulative Grade Point Average (scored out of 10)
- **Research**: Research experience (0 or 1)
- **Chance of Admit**: Chance of admission (scored out of 1)

Bon travail!
