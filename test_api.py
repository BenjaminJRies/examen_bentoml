"""
Test script for the BentoML Admission Prediction API
This script tests the authentication and prediction endpoints.
"""

import requests
import json
import pandas as pd
from typing import Dict, Any

# API configuration
API_BASE_URL = "http://localhost:3000"  # Default BentoML port
HEADERS = {"Content-Type": "application/json"}

# Test credentials
TEST_USERS = {
    "admin": "admin123",
    "user": "user123",
    "test": "test123"
}

# Sample student data for testing
SAMPLE_STUDENTS = [
    {
        "GRE_Score": 337,
        "TOEFL_Score": 118,
        "University_Rating": 4,
        "SOP": 4.5,
        "LOR": 4.5,
        "CGPA": 9.65,
        "Research": 1
    },
    {
        "GRE_Score": 324,
        "TOEFL_Score": 107,
        "University_Rating": 4,
        "SOP": 4.0,
        "LOR": 4.5,
        "CGPA": 8.87,
        "Research": 1
    },
    {
        "GRE_Score": 316,
        "TOEFL_Score": 104,
        "University_Rating": 3,
        "SOP": 3.0,
        "LOR": 3.5,
        "CGPA": 8.0,
        "Research": 1
    },
    {
        "GRE_Score": 300,
        "TOEFL_Score": 95,
        "University_Rating": 2,
        "SOP": 2.5,
        "LOR": 3.0,
        "CGPA": 7.2,
        "Research": 0
    }
]

class APITester:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.access_token = None
        
    def test_health_check(self) -> bool:
        """Test the health check endpoint (no auth required)"""
        print("🏥 Testing Health Check Endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health check passed!")
                print(f"   Service: {health_data.get('service')}")
                print(f"   Status: {health_data.get('status')}")
                print(f"   Version: {health_data.get('version')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_login(self, username: str = "test", password: str = "test123") -> bool:
        """Test login endpoint and store access token"""
        print(f"🔐 Testing Login with {username}...")
        try:
            login_data = {
                "username": username,
                "password": password
            }
            response = requests.post(
                f"{self.base_url}/login",
                json=login_data,
                headers=HEADERS
            )
            
            if response.status_code == 200:
                auth_data = response.json()
                self.access_token = auth_data["access_token"]
                print("✅ Login successful!")
                print(f"   Token type: {auth_data.get('token_type')}")
                print(f"   Expires in: {auth_data.get('expires_in')} seconds")
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def test_single_prediction(self, student_data: Dict[str, Any]) -> bool:
        """Test single prediction endpoint"""
        print("🎯 Testing Single Prediction...")
        if not self.access_token:
            print("❌ No access token available. Please login first.")
            return False
        
        try:
            auth_headers = {
                **HEADERS,
                "Authorization": f"Bearer {self.access_token}"
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                json=student_data,
                headers=auth_headers
            )
            
            if response.status_code == 200:
                prediction = response.json()
                print("✅ Single prediction successful!")
                print(f"   Student: GRE {student_data['GRE_Score']}, TOEFL {student_data['TOEFL_Score']}, CGPA {student_data['CGPA']}")
                print(f"   Chance of Admit: {prediction['chance_of_admit']:.4f} ({prediction['chance_of_admit']*100:.2f}%)")
                print(f"   Confidence: {prediction['confidence_level']}")
                print(f"   Interpretation: {prediction['interpretation']}")
                return True
            else:
                print(f"❌ Single prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Single prediction error: {e}")
            return False
    
    def test_batch_prediction(self, students_list: list) -> bool:
        """Test batch prediction endpoint"""
        print("📊 Testing Batch Prediction...")
        if not self.access_token:
            print("❌ No access token available. Please login first.")
            return False
        
        try:
            auth_headers = {
                **HEADERS,
                "Authorization": f"Bearer {self.access_token}"
            }
            
            batch_data = {"students": students_list}
            response = requests.post(
                f"{self.base_url}/predict_batch",
                json=batch_data,
                headers=auth_headers
            )
            
            if response.status_code == 200:
                batch_result = response.json()
                print("✅ Batch prediction successful!")
                print(f"   Total students: {batch_result['summary']['total_students']}")
                print(f"   Average admission chance: {batch_result['summary']['average_admission_chance']:.4f}")
                print(f"   Highest chance: {batch_result['summary']['highest_chance']:.4f}")
                print(f"   Lowest chance: {batch_result['summary']['lowest_chance']:.4f}")
                print(f"   High confidence predictions: {batch_result['summary']['high_confidence_count']}")
                
                print("\n   Individual predictions:")
                for i, pred in enumerate(batch_result['predictions'][:3]):  # Show first 3
                    student = students_list[i]
                    print(f"   Student {i+1}: {pred['chance_of_admit']:.4f} ({pred['confidence_level']})")
                    print(f"      GRE: {student['GRE_Score']}, CGPA: {student['CGPA']}")
                
                return True
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return False
    
    def test_unauthorized_access(self) -> bool:
        """Test that endpoints require authentication"""
        print("🚫 Testing Unauthorized Access...")
        try:
            # Try prediction without token
            response = requests.post(
                f"{self.base_url}/predict",
                json=SAMPLE_STUDENTS[0],
                headers=HEADERS
            )
            
            if response.status_code == 401:
                print("✅ Unauthorized access properly blocked!")
                return True
            else:
                print(f"❌ Expected 401, got {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Unauthorized access test error: {e}")
            return False
    
    def test_invalid_credentials(self) -> bool:
        """Test login with invalid credentials"""
        print("🔒 Testing Invalid Credentials...")
        try:
            login_data = {
                "username": "invalid_user",
                "password": "invalid_password"
            }
            response = requests.post(
                f"{self.base_url}/login",
                json=login_data,
                headers=HEADERS
            )
            
            if response.status_code == 401:
                print("✅ Invalid credentials properly rejected!")
                return True
            else:
                print(f"❌ Expected 401, got {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Invalid credentials test error: {e}")
            return False
    
    def run_comprehensive_tests(self) -> None:
        """Run all tests"""
        print("🧪 STARTING COMPREHENSIVE API TESTS")
        print("=" * 60)
        
        test_results = []
        
        # Test 1: Health check
        test_results.append(("Health Check", self.test_health_check()))
        
        # Test 2: Invalid credentials
        test_results.append(("Invalid Credentials", self.test_invalid_credentials()))
        
        # Test 3: Unauthorized access
        test_results.append(("Unauthorized Access", self.test_unauthorized_access()))
        
        # Test 4: Valid login
        test_results.append(("Valid Login", self.test_login()))
        
        # Test 5: Single prediction
        test_results.append(("Single Prediction", self.test_single_prediction(SAMPLE_STUDENTS[0])))
        
        # Test 6: Batch prediction
        test_results.append(("Batch Prediction", self.test_batch_prediction(SAMPLE_STUDENTS)))
        
        # Test 7: Test different student profiles
        print("\n🎓 Testing Different Student Profiles...")
        high_performer = SAMPLE_STUDENTS[0]  # Strong profile
        low_performer = SAMPLE_STUDENTS[3]   # Weak profile
        
        test_results.append(("High Performer", self.test_single_prediction(high_performer)))
        test_results.append(("Low Performer", self.test_single_prediction(low_performer)))
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = 0
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<20}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(test_results)} tests passed")
        if passed == len(test_results):
            print("🎉 All tests passed! API is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the API.")

def main():
    """Main function to run tests"""
    print("🚀 BentoML Admission Prediction API Tester")
    print("=" * 60)
    print("This script will test the API endpoints.")
    print("Make sure the BentoML service is running on http://localhost:3000")
    print()
    
    # Check if user wants to continue
    response = input("Press Enter to start tests or 'q' to quit: ")
    if response.lower() == 'q':
        return
    
    # Create tester instance and run tests
    tester = APITester()
    tester.run_comprehensive_tests()

if __name__ == "__main__":
    main()
