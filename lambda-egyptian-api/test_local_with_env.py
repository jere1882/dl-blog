#!/usr/bin/env python3
"""
Local test script that loads environment variables and tests the Lambda function
"""
import json
import base64
import os
from pathlib import Path

def load_env_file(env_file_path):
    """Load environment variables from a .env file"""
    if not os.path.exists(env_file_path):
        print(f"Environment file {env_file_path} not found")
        return
    
    with open(env_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
                print(f"Loaded: {key.strip()}")

def test_lambda_function():
    """Test the Lambda function locally with environment variables"""
    
    # Load environment variables
    env_file = Path(__file__).parent / "env.local"
    load_env_file(env_file)
    
    # Check if Google API key is loaded
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    if not google_api_key or google_api_key == 'your-actual-google-api-key-here':
        print("❌ Please set your actual Google API key in env.local")
        print("   Replace 'your-actual-google-api-key-here' with your real API key")
        return
    
    print("✅ Google API key loaded successfully")
    
    # Create a minimal test image (1x1 pixel PNG)
    test_image_data = base64.b64encode(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x07tIME\x07\xe7\x01\x01\x00\x01\x1d\xf5\xc4\xd8\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82').decode('utf-8')
    
    # Create test event
    test_event = {
        'httpMethod': 'POST',
        'body': json.dumps({
            'image': test_image_data,
            'speed': 'fast',
            'imageType': 'unknown'
        })
    }
    
    print("Testing Lambda function locally...")
    print(f"Event: {json.dumps(test_event, indent=2)}")
    
    try:
        # Import and test the Lambda function
        import sys
        sys.path.append(str(Path(__file__).parent))
        
        from lambda_function import lambda_handler
        
        response = lambda_handler(test_event, None)
        print(f"\nResponse: {json.dumps(response, indent=2)}")
        
        if response['statusCode'] == 200:
            print("\n✅ Test passed! Function is working correctly.")
        else:
            print(f"\n❌ Test failed with status code: {response['statusCode']}")
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lambda_function()
