"""
Test script for the Student Admission Prediction API
Based on the accidents project pattern
"""

import requests
import json
import time
from typing import Dict, Any

class AdmissionAPITester:
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
    
    def test_health(self) -> bool:
        """Test the health endpoint"""
        print("🏥 Testing Health Endpoint...")
        try:
            response = self.session.post(f"{self.base_url}/health", json={})
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_login(self, username: str = "test", password: str = "test123") -> bool:
        """Test the login endpoint"""
        print(f"🔐 Testing Login Endpoint (user: {username})...")
        try:
            login_data = {
                "username": username,
                "password": password
            }
            response = self.session.post(f"{self.base_url}/login", json=login_data)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    self.token = data.get("access_token")
                    print(f"✅ Login successful!")
                    print(f"   Token: {self.token[:50]}...")
                    print(f"   User: {data.get('user', {}).get('username')}")
                    return True
                else:
                    print(f"❌ Login failed: {data}")
                    return False
            else:
                print(f"❌ Login failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Login error: {e}")
            return False
    
    def test_predict(self) -> bool:
        """Test the predict endpoint"""
        print("🎯 Testing Predict Endpoint...")
        if not self.token:
            print("❌ No token available. Please login first.")
            return False
        
        try:
            # Sample student data
            predict_data = {
                "GRE_Score": 320,
                "TOEFL_Score": 110,
                "University_Rating": 4,
                "SOP": 4.5,
                "LOR": 4.0,
                "CGPA": 8.5,
                "Research": 1
            }
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/models/admission_predictor/predict", 
                json=predict_data,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    print(f"✅ Prediction successful!")
                    print(f"   Chance of Admission: {data.get('chance_of_admit', 'N/A')}")
                    print(f"   Confidence Level: {data.get('confidence_level', 'N/A')}")
                    print(f"   Interpretation: {data.get('interpretation', 'N/A')}")
                    print(f"   User: {data.get('user', 'N/A')}")
                    return True
                else:
                    print(f"❌ Prediction failed: {data}")
                    return False
            else:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Prediction error: {e}")
            return False
    
    def test_predict_batch(self) -> bool:
        """Test the batch predict endpoint"""
        print("📊 Testing Batch Predict Endpoint...")
        if not self.token:
            print("❌ No token available. Please login first.")
            return False
        
        try:
            # Sample batch data
            batch_data = {
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
                        "GRE_Score": 300,
                        "TOEFL_Score": 95,
                        "University_Rating": 3,
                        "SOP": 3.5,
                        "LOR": 3.5,
                        "CGPA": 7.5,
                        "Research": 0
                    },
                    {
                        "GRE_Score": 310,
                        "TOEFL_Score": 100,
                        "University_Rating": 4,
                        "SOP": 4.0,
                        "LOR": 4.5,
                        "CGPA": 8.0,
                        "Research": 1
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/models/admission_predictor/predict_batch", 
                json=batch_data,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    print(f"✅ Batch prediction successful!")
                    predictions = data.get("predictions", [])
                    summary = data.get("summary", {})
                    
                    print(f"   Total Students: {summary.get('total_students', 'N/A')}")
                    print(f"   Successful Predictions: {summary.get('successful_predictions', 'N/A')}")
                    print(f"   Average Chance: {summary.get('average_chance', 'N/A')}")
                    print(f"   Range: {summary.get('min_chance', 'N/A')} - {summary.get('max_chance', 'N/A')}")
                    print(f"   User: {data.get('user', 'N/A')}")
                    
                    print("\\n   Individual Predictions:")
                    for pred in predictions[:3]:  # Show first 3
                        print(f"   Student {pred.get('student_id')}: {pred.get('chance_of_admit')} ({pred.get('confidence_level')})")
                    
                    return True
                else:
                    print(f"❌ Batch prediction failed: {data}")
                    return False
            else:
                print(f"❌ Batch prediction failed: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Batch prediction error: {e}")
            return False
    
    def test_unauthorized_access(self) -> bool:
        """Test unauthorized access"""
        print("🚫 Testing Unauthorized Access...")
        try:
            # Try to predict without token
            predict_data = {
                "GRE_Score": 320,
                "TOEFL_Score": 110,
                "University_Rating": 4,
                "SOP": 4.5,
                "LOR": 4.0,
                "CGPA": 8.5,
                "Research": 1
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/models/admission_predictor/predict", 
                json=predict_data
            )
            
            if response.status_code == 401:
                print("✅ Unauthorized access properly blocked!")
                return True
            else:
                print(f"❌ Unauthorized access not properly handled: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Unauthorized access test error: {e}")
            return False
    
    def test_invalid_credentials(self) -> bool:
        """Test invalid login credentials"""
        print("🔒 Testing Invalid Credentials...")
        try:
            login_data = {
                "username": "invalid_user",
                "password": "wrong_password"
            }
            response = self.session.post(f"{self.base_url}/login", json=login_data)
            
            if response.status_code == 401:
                print("✅ Invalid credentials properly rejected!")
                return True
            else:
                print(f"❌ Invalid credentials not properly handled: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Invalid credentials test error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("🚀 STARTING API TESTS - BENTOML ADMISSION PREDICTION SERVICE")
        print("=" * 70)
        
        results = {}
        
        # Test health endpoint
        results["health"] = self.test_health()
        print()
        
        # Test invalid credentials
        results["invalid_credentials"] = self.test_invalid_credentials()
        print()
        
        # Test login
        results["login"] = self.test_login()
        print()
        
        # Test unauthorized access
        results["unauthorized"] = self.test_unauthorized_access()
        print()
        
        # Test predict (requires login)
        if results["login"]:
            results["predict"] = self.test_predict()
            print()
            
            # Test batch predict
            results["predict_batch"] = self.test_predict_batch()
            print()
        else:
            results["predict"] = False
            results["predict_batch"] = False
            print("⏭️  Skipping prediction tests (login failed)")
        
        # Print summary
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 70)
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{test_name.upper():<20} {status}")
        
        print(f"\\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️  Some tests failed. Check the API service.")
        
        return results

def main():
    """Main test function"""
    print("🧪 Student Admission Prediction API - BentoML Test Suite")
    print("=" * 70)
    
    # Test with different base URLs if needed
    base_urls = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    for base_url in base_urls:
        print(f"\\n🌐 Testing API at: {base_url}")
        tester = AdmissionAPITester(base_url)
        
        # Quick health check to see if API is running
        if tester.test_health():
            print(f"✅ API is running at {base_url}")
            results = tester.run_all_tests()
            break
        else:
            print(f"❌ API not accessible at {base_url}")
    else:
        print("\\n❌ API is not running at any of the tested URLs.")
        print("Please start the BentoML service first:")
        print("   bentoml serve src.service_new:service --reload --port 3000")

if __name__ == "__main__":
    main()
