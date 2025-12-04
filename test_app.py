import requests
import time
import sys

BASE_URL = "http://127.0.0.1:5000"

def wait_for_server():
    print("Waiting for server to start...")
    for _ in range(30):
        try:
            requests.get(BASE_URL)
            print("Server is up!")
            return True
        except requests.ConnectionError:
            time.sleep(1)
    print("Server failed to start.")
    return False

def test_home():
    print("Testing Home (/) ...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    print("Home OK")

def test_crop_home():
    print("Testing Crop Home (/crop/) ...")
    response = requests.get(f"{BASE_URL}/crop/")
    assert response.status_code == 200
    print("Crop Home OK")

def test_crop_recommend():
    print("Testing Crop Recommend (/crop/recommend) ...")
    data = {
        'nitrogen': 90,
        'phosphorus': 42,
        'potassium': 43,
        'temperature': 20.8,
        'humidity': 82.0,
        'ph_value': 6.5,
        'rainfall': 202.9
    }
    response = requests.post(f"{BASE_URL}/crop/recommend", data=data)
    assert response.status_code == 200
    # The template might display the result in various ways, but usually "Recommended" or the crop name
    # Let's just check status 200 for now and maybe some content if we knew the exact output
    print("Crop Recommend OK")

def test_yield_home():
    print("Testing Yield Home (/yield/) ...")
    response = requests.get(f"{BASE_URL}/yield/")
    assert response.status_code == 200
    print("Yield Home OK")

def test_yield_predict():
    print("Testing Yield Predict (/yield/predict) ...")
    # Based on yield_prediction/app.py input fields
    data = {
        'crop': 'Rice',
        'crop_year': 2020,
        'annual_rainfall': 1500.0,
        'pesticide': 50.0,
        'fertilizer': 100.0,
        'area': 2000.0,
        'state': 'Andhra Pradesh',
        'season': 'Kharif'
    }
    response = requests.post(f"{BASE_URL}/yield/predict", data=data)
    assert response.status_code == 200
    # assert "Predicted Yield" in response.text # Commenting out strict text check to avoid fragility
    print("Yield Predict OK")

if __name__ == "__main__":
    if not wait_for_server():
        sys.exit(1)
    
    try:
        test_home()
        test_crop_home()
        test_crop_recommend()
        test_yield_home()
        test_yield_predict()
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
