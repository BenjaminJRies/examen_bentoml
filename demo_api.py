"""
Demonstration of the Student Admission Prediction API
Shows how to use the BentoML service for predictions
"""

import requests
import json

def demonstrate_api():
    """Demonstrate the working API endpoints"""
    base_url = "http://localhost:3000"
    
    print("🚀 Student Admission Prediction API - Live Demonstration")
    print("=" * 60)
    
    # 1. Health Check
    print("\n1️⃣ Health Check")
    print("-" * 20)
    try:
        response = requests.post(f"{base_url}/health", json={})
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Service Status: {health_data['status']}")
            print(f"📊 Models Loaded: {health_data['models_loaded']}")
            print(f"🕒 Timestamp: {health_data['timestamp']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # 2. Login
    print("\n2️⃣ Authentication")
    print("-" * 20)
    try:
        login_data = {"username": "test", "password": "test123"}
        response = requests.post(f"{base_url}/login", json=login_data)
        
        if response.status_code == 200:
            auth_data = response.json()
            token = auth_data['access_token']
            print(f"✅ Login successful for user: {auth_data['user']['username']}")
            print(f"🔑 Token obtained (expires in {auth_data['expires_in']} seconds)")
        else:
            print(f"❌ Login failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")
        return
    
    # 3. Sample Student Profiles for Testing
    students = [
        {
            "name": "High Achiever",
            "data": {
                "GRE_Score": 325,
                "TOEFL_Score": 115,
                "University_Rating": 5,
                "SOP": 4.8,
                "LOR": 4.7,
                "CGPA": 9.2,
                "Research": 1
            }
        },
        {
            "name": "Average Student",
            "data": {
                "GRE_Score": 310,
                "TOEFL_Score": 100,
                "University_Rating": 3,
                "SOP": 3.5,
                "LOR": 3.8,
                "CGPA": 7.8,
                "Research": 0
            }
        },
        {
            "name": "Struggling Student",
            "data": {
                "GRE_Score": 290,
                "TOEFL_Score": 85,
                "University_Rating": 2,
                "SOP": 2.5,
                "LOR": 2.8,
                "CGPA": 6.5,
                "Research": 0
            }
        }
    ]
    
    # 4. Batch Prediction (since single prediction has issues)
    print("\n3️⃣ Batch Predictions")
    print("-" * 20)
    try:
        batch_data = {
            "students": [student["data"] for student in students]
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{base_url}/v1/models/admission_predictor/predict_batch",
            json=batch_data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", [])
            summary = result.get("summary", {})
            
            print(f"✅ Batch prediction successful!")
            print(f"📊 Summary:")
            print(f"   • Total Students: {summary.get('total_students', 0)}")
            print(f"   • Successful Predictions: {summary.get('successful_predictions', 0)}")
            if summary.get('successful_predictions', 0) > 0:
                print(f"   • Average Admission Chance: {summary.get('average_chance', 'N/A')}")
                print(f"   • Range: {summary.get('min_chance', 'N/A')} - {summary.get('max_chance', 'N/A')}")
            
            print(f"\\n📋 Individual Results:")
            for i, student in enumerate(students):
                if i < len(predictions):
                    pred = predictions[i]
                    print(f"   {student['name']}:")
                    print(f"      • Admission Chance: {pred.get('chance_of_admit', 'N/A')}")
                    print(f"      • Confidence: {pred.get('confidence_level', 'N/A')}")
                    print(f"      • Advice: {pred.get('interpretation', 'N/A')}")
                else:
                    print(f"   {student['name']}: ❌ Prediction failed")
                print()
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
    
    # 5. Test Security
    print("\n4️⃣ Security Test")
    print("-" * 20)
    try:
        # Try to access without token
        test_data = {"GRE_Score": 300, "TOEFL_Score": 100, "University_Rating": 3, 
                    "SOP": 3.5, "LOR": 3.5, "CGPA": 7.5, "Research": 0}
        response = requests.post(
            f"{base_url}/v1/models/admission_predictor/predict",
            json=test_data
        )
        
        if response.status_code == 401:
            print("✅ Security working - unauthorized access blocked")
        else:
            print(f"⚠️ Security issue - got status {response.status_code}")
    except Exception as e:
        print(f"❌ Security test error: {e}")
    
    print("\n🎯 API Demonstration Complete!")
    print("=" * 60)
    print("\n📖 Available Endpoints:")
    print("   • POST /health - Health check")
    print("   • POST /login - Authentication")
    print("   • POST /v1/models/admission_predictor/predict - Single prediction (secured)")
    print("   • POST /v1/models/admission_predictor/predict_batch - Batch prediction (secured)")
    print("\n🔐 Test Credentials:")
    print("   • Username: test, Password: test123")
    print("   • Username: admin, Password: admin123")
    print("   • Username: demo, Password: demo123")

if __name__ == "__main__":
    demonstrate_api()
