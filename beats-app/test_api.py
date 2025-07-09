import requests
import os
import time

def test_health_check():
    """Test the health check endpoint"""
    url = "http://localhost:5015/api/health"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful!")
            print(f"Status: {result['status']}")
            print(f"Model loaded: {result['model_loaded']}")
            print(f"Device: {result['device']}")
            return True
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the Flask app is running on localhost:5015")
        return False
    except Exception as e:
        print(f"❌ Error during health check: {str(e)}")
        return False

def test_prediction_api():
    """Test the prediction API endpoint"""
    
    # First check if server is healthy
    if not test_health_check():
        print("\nSkipping prediction test due to health check failure.")
        return
    
    # URL of your Flask app
    url = "http://localhost:5015/api/predict"
    
    # Check if there's a test audio file in the current directory
    test_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if not test_files:
        print("\nNo .wav files found in current directory for testing.")
        print("Please place a .wav file in the current directory to test the API.")
        return
    
    # Use the first .wav file found
    test_file = test_files[0]
    print(f"\nTesting with file: {test_file}")
    
    try:
        # Prepare the file for upload
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'audio/wav')}
            
            # Make the request with timeout
            response = requests.post(url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API test successful!")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Real probability: {result['probabilities']['real']:.4f}")
                print(f"Fake probability: {result['probabilities']['fake']:.4f}")
            else:
                print(f"❌ API test failed with status code: {response.status_code}")
                print(f"Error: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the Flask app is running on localhost:5015")
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The model might be taking too long to process.")
    except Exception as e:
        print(f"❌ Error during API test: {str(e)}")

def main():
    print("Testing BEATs Audio Deepfake Detection API")
    print("=" * 50)
    
    # Wait a moment for server to start if needed
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    test_prediction_api()

if __name__ == "__main__":
    main() 