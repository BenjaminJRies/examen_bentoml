"""
Unit Tests for BentoML Admission Prediction Service
As required by exam section 1.5

Tests cover:
1. JWT Authentication Test
2. Login API Test  
3. Prediction API Test
"""

import unittest
import requests
import json
import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:3000"
HEADERS = {"Content-Type": "application/json"}

# JWT Configuration (from service)
JWT_SECRET_KEY = "admission_prediction_secret_key_2025"
JWT_ALGORITHM = "HS256"

# Test credentials (from service)
VALID_USER = {"username": "test", "password": "test123"}
INVALID_USER = {"username": "invalid", "password": "wrong"}

# Sample test data
VALID_STUDENT_DATA = {
    "GRE_Score": 337,
    "TOEFL_Score": 118,
    "University_Rating": 4,
    "SOP": 4.5,
    "LOR": 4.5,
    "CGPA": 9.65,
    "Research": 1
}

INVALID_STUDENT_DATA = {
    "GRE_Score": "invalid",  # Should be numeric
    "TOEFL_Score": 118,
    "University_Rating": 4,
    "SOP": 4.5,
    "LOR": 4.5,
    "CGPA": 9.65,
    "Research": 1
}


class TestJWTAuthentication(unittest.TestCase):
    """Test JWT Authentication functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Get a valid token for testing
        response = requests.post(
            f"{API_BASE_URL}/login",
            json=VALID_USER,
            headers=HEADERS
        )
        if response.status_code == 200:
            self.valid_token = response.json()["access_token"]
        else:
            self.skipTest("Cannot get valid token - service may not be running")
    
    def test_missing_jwt_token(self):
        """Verify that authentication fails if JWT token is missing"""
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=VALID_STUDENT_DATA,
            headers=HEADERS  # No Authorization header
        )
        
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing authentication token", response.json().get("detail", ""))
    
    def test_invalid_jwt_token(self):
        """Verify that authentication fails if JWT token is invalid"""
        invalid_headers = {
            **HEADERS,
            "Authorization": "Bearer invalid_token_here"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=VALID_STUDENT_DATA,
            headers=invalid_headers
        )
        
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid token", response.json().get("detail", ""))
    
    def test_expired_jwt_token(self):
        """Verify that authentication fails if JWT token has expired"""
        # Create an expired token
        expired_payload = {
            "sub": "test",
            "exp": datetime.utcnow() - timedelta(seconds=1)  # Expired 1 second ago
        }
        expired_token = jwt.encode(expired_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        expired_headers = {
            **HEADERS,
            "Authorization": f"Bearer {expired_token}"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=VALID_STUDENT_DATA,
            headers=expired_headers
        )
        
        self.assertEqual(response.status_code, 401)
        self.assertIn("Token has expired", response.json().get("detail", ""))
    
    def test_valid_jwt_token(self):
        """Verify that authentication succeeds with a valid JWT token"""
        valid_headers = {
            **HEADERS,
            "Authorization": f"Bearer {self.valid_token}"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=VALID_STUDENT_DATA,
            headers=valid_headers
        )
        
        # Should not be 401 (authentication should succeed)
        self.assertNotEqual(response.status_code, 401)
        # Should return successful prediction (200)
        self.assertEqual(response.status_code, 200)


class TestLoginAPI(unittest.TestCase):
    """Test Login API functionality"""
    
    def test_valid_credentials_return_jwt_token(self):
        """Verify that API returns a valid JWT token for correct credentials"""
        response = requests.post(
            f"{API_BASE_URL}/login",
            json=VALID_USER,
            headers=HEADERS
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("access_token", data)
        self.assertIn("token_type", data)
        self.assertIn("expires_in", data)
        self.assertEqual(data["token_type"], "bearer")
        
        # Verify the token is valid JWT
        token = data["access_token"]
        try:
            decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            self.assertEqual(decoded["sub"], VALID_USER["username"])
        except jwt.InvalidTokenError:
            self.fail("Returned token is not a valid JWT")
    
    def test_invalid_credentials_return_401(self):
        """Verify that API returns 401 error for incorrect credentials"""
        response = requests.post(
            f"{API_BASE_URL}/login",
            json=INVALID_USER,
            headers=HEADERS
        )
        
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid credentials", response.json().get("detail", ""))
    
    def test_missing_credentials_return_error(self):
        """Verify that API returns error for missing credentials"""
        # Test with missing password
        incomplete_user = {"username": "test"}
        response = requests.post(
            f"{API_BASE_URL}/login",
            json=incomplete_user,
            headers=HEADERS
        )
        
        self.assertIn(response.status_code, [400, 422])  # Bad request or validation error
    
    def test_empty_credentials_return_401(self):
        """Verify that API returns 401 for empty credentials"""
        empty_user = {"username": "", "password": ""}
        response = requests.post(
            f"{API_BASE_URL}/login",
            json=empty_user,
            headers=HEADERS
        )
        
        self.assertEqual(response.status_code, 401)


class TestPredictionAPI(unittest.TestCase):
    """Test Prediction API functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Get a valid token for testing
        response = requests.post(
            f"{API_BASE_URL}/login",
            json=VALID_USER,
            headers=HEADERS
        )
        if response.status_code == 200:
            self.valid_token = response.json()["access_token"]
            self.auth_headers = {
                **HEADERS,
                "Authorization": f"Bearer {self.valid_token}"
            }
        else:
            self.skipTest("Cannot get valid token - service may not be running")
    
    def test_prediction_without_jwt_returns_401(self):
        """Verify that API returns 401 if JWT token is missing"""
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=VALID_STUDENT_DATA,
            headers=HEADERS  # No authorization
        )
        
        self.assertEqual(response.status_code, 401)
    
    def test_prediction_with_invalid_jwt_returns_401(self):
        """Verify that API returns 401 if JWT token is invalid"""
        invalid_headers = {
            **HEADERS,
            "Authorization": "Bearer invalid_token"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=VALID_STUDENT_DATA,
            headers=invalid_headers
        )
        
        self.assertEqual(response.status_code, 401)
    
    def test_valid_prediction_returns_result(self):
        """Verify that API returns valid prediction for correct input data"""
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=VALID_STUDENT_DATA,
            headers=self.auth_headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("chance_of_admit", data)
        self.assertIn("confidence_level", data)
        self.assertIn("interpretation", data)
        
        # Verify prediction is in valid range [0, 1]
        chance = data["chance_of_admit"]
        self.assertGreaterEqual(chance, 0.0)
        self.assertLessEqual(chance, 1.0)
        
        # Verify confidence level exists
        self.assertIn(data["confidence_level"], ["High", "Medium", "Low"])
    
    def test_invalid_input_data_returns_error(self):
        """Verify that API returns error for invalid input data"""
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=INVALID_STUDENT_DATA,
            headers=self.auth_headers
        )
        
        # Should return validation error (400 or 422)
        self.assertIn(response.status_code, [400, 422])
    
    def test_missing_required_fields_returns_error(self):
        """Verify that API returns error for missing required fields"""
        incomplete_data = {
            "GRE_Score": 337,
            "TOEFL_Score": 118
            # Missing other required fields
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=incomplete_data,
            headers=self.auth_headers
        )
        
        # Should return validation error
        self.assertIn(response.status_code, [400, 422])
    
    def test_batch_prediction_functionality(self):
        """Test batch prediction endpoint (bonus feature)"""
        batch_data = {
            "students": [VALID_STUDENT_DATA, VALID_STUDENT_DATA]
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict_batch",
            json=batch_data,
            headers=self.auth_headers
        )
        
        if response.status_code == 200:
            # If batch endpoint exists, verify response structure
            data = response.json()
            self.assertIn("predictions", data)
            self.assertIn("summary", data)
            self.assertEqual(len(data["predictions"]), 2)
        else:
            # If endpoint doesn't exist, that's also acceptable
            self.assertIn(response.status_code, [404, 405])


class TestHealthEndpoint(unittest.TestCase):
    """Test Health endpoint functionality"""
    
    def test_health_endpoint_accessible(self):
        """Verify that health endpoint is accessible without authentication"""
        response = requests.post(f"{API_BASE_URL}/health", json={})
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")


class TestServiceIntegration(unittest.TestCase):
    """Integration tests for the complete service"""
    
    def test_complete_workflow(self):
        """Test complete workflow: login -> predict -> logout"""
        # Step 1: Login
        login_response = requests.post(
            f"{API_BASE_URL}/login",
            json=VALID_USER,
            headers=HEADERS
        )
        
        self.assertEqual(login_response.status_code, 200)
        token = login_response.json()["access_token"]
        
        # Step 2: Make prediction
        auth_headers = {
            **HEADERS,
            "Authorization": f"Bearer {token}"
        }
        
        prediction_response = requests.post(
            f"{API_BASE_URL}/v1/models/admission_predictor/predict",
            json=VALID_STUDENT_DATA,
            headers=auth_headers
        )
        
        self.assertEqual(prediction_response.status_code, 200)
        
        # Step 3: Verify prediction result
        prediction_data = prediction_response.json()
        self.assertIn("chance_of_admit", prediction_data)
        
        # The prediction should be reasonable for a high-performing student
        chance = prediction_data["chance_of_admit"]
        self.assertGreater(chance, 0.5)  # High GRE, TOEFL, CGPA should give good chance


def run_all_tests():
    """Run all unit tests"""
    print("🧪 RUNNING UNIT TESTS FOR BENTOML ADMISSION PREDICTION SERVICE")
    print("=" * 70)
    print("Testing requirements from exam section 1.5:")
    print("1. JWT Authentication Test")
    print("2. Login API Test")
    print("3. Prediction API Test")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestJWTAuthentication,
        TestLoginAPI,
        TestPredictionAPI,
        TestHealthEndpoint,
        TestServiceIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, failure in result.failures:
            print(f"   {test}: {failure}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, error in result.errors:
            print(f"   {test}: {error}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ JWT Authentication: Working correctly")
        print("✅ Login API: Working correctly")
        print("✅ Prediction API: Working correctly")
        print("✅ Service Integration: Working correctly")
        print("\n🏆 Your BentoML service meets all exam requirements!")
    
    return result


if __name__ == "__main__":
    try:
        # Check if service is running
        health_response = requests.post(f"{API_BASE_URL}/health", json={}, timeout=5)
        if health_response.status_code != 200:
            print("❌ Service is not running on http://localhost:3000")
            print("Please start your BentoML service first:")
            print("   docker run -p 3000:3000 benjaminries_admission_prediction")
            exit(1)
        else:
            print("✅ Service is running, starting tests...")
            run_all_tests()
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to service on http://localhost:3000")
        print("Please make sure your Docker container is running:")
        print("   docker run -p 3000:3000 benjaminries_admission_prediction")
        exit(1)
