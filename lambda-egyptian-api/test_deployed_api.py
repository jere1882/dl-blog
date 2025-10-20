#!/usr/bin/env python3
"""
Test the deployed Lambda function via API Gateway
"""
import json
import base64
import requests

def test_deployed_function():
    """Test the deployed function with a real Egyptian image"""
    
    # Read the base64 encoded image
    with open('test_image_base64.txt', 'r') as f:
        image_data = f.read().strip()
    
    print(f"Image data size: {len(image_data)} characters (base64)")
    
    # Test payload for Egyptian art analysis
    payload = {
        'image': image_data,
        'speed': 'fast',
        'imageType': 'tomb'  # This is likely a tomb image from Valley of Kings
    }
    
    print("Testing deployed Lambda function via API Gateway...")
    print(f"Payload keys: {list(payload.keys())}")
    
    # API Gateway endpoint
    url = "https://l7tp0pix83.execute-api.us-east-2.amazonaws.com/default/decipher-egyptian-hieroglyphs"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "SL1KJ4ShwF37CA6jRcISy32kJgx93NA7Kc9Xj8j2"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Function is working!")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (60 seconds)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_deployed_function()

