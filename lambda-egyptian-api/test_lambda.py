#!/usr/bin/env python3
"""
Test script to verify the Lambda function works with a real API call
"""
import json
import base64
import requests
import os
from pathlib import Path

def test_lambda_api():
    """Test the deployed Lambda function"""
    
    # Configuration
    lambda_url = os.environ.get('LAMBDA_URL', 'https://l7tp0pix83.execute-api.us-east-2.amazonaws.com/default/egyptian-art-analyzer')
    
    # Create a test image (1x1 pixel PNG)
    test_image_data = base64.b64encode(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x07tIME\x07\xe7\x01\x01\x00\x01\x1d\xf5\xc4\xd8\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82').decode('utf-8')
    
    # Test payload
    payload = {
        'image': test_image_data,
        'speed': 'fast',
        'imageType': 'unknown'
    }
    
    print(f"Testing Lambda API at: {lambda_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Get API key from environment or use a placeholder
        api_key = os.environ.get('API_KEY', 'SL1KJ4ShwF37CA6jRcISy32kJgx93NA7Kc9Xj8j2')
        
        response = requests.post(
            lambda_url,
            headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key
            },
            json=payload,
            timeout=30
        )
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success! Response: {json.dumps(result, indent=2)}")
        else:
            print(f"\n❌ Error! Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Request failed: {str(e)}")

if __name__ == "__main__":
    test_lambda_api()
