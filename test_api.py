"""
Simple test script for the Flask API
Run this after starting the API server to verify it works correctly.
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\nTesting prediction endpoint...")
    
    # Sample input data
    test_data = {
        "A1_Score": 1,
        "A2_Score": 0,
        "A3_Score": 1,
        "A4_Score": 0,
        "A5_Score": 1,
        "A6_Score": 0,
        "A7_Score": 1,
        "A8_Score": 0,
        "A9_Score": 1,
        "A10_Score": 1,
        "age": 25,
        "gender": "m",
        "ethnicity": "White-European",
        "jaundice": "no",
        "austim": "no",
        "contry_of_res": "United States",
        "used_app_before": "no",
        "result": 6.5,
        "relation": "Self"
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=test_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("API Test Script")
    print("=" * 50)
    
    # Test health check
    health_ok = test_health_check()
    
    if health_ok:
        # Test prediction
        prediction_ok = test_prediction()
        
        if prediction_ok:
            print("\n" + "=" * 50)
            print("✅ All tests passed!")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("❌ Prediction test failed!")
            print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ Health check failed! Make sure the API is running.")
        print("=" * 50)
        print("\nTo start the API, run: python APIs/app.py")

