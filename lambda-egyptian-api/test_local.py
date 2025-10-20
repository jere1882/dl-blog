#!/usr/bin/env python3
"""
Simple local test for the Lambda function
Run this from within the lambda-egyptian-api directory
"""
import json
import base64
import os
import io
from pathlib import Path
from PIL import Image

def load_env_file():
    """Load environment variables from env.local"""
    env_file = Path("env.local")
    if not env_file.exists():
        print("❌ env.local file not found")
        print("   Please create env.local with your Google API key")
        return False
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
                print(f"✅ Loaded: {key.strip()}")
    
    return True

def test_lambda():
    """Test the Lambda function locally"""
    print("🧪 Testing Lambda function locally...")
    
    # Load environment variables
    if not load_env_file():
        return
    
    # Check if Google API key is set
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    if not google_api_key or google_api_key == 'your-actual-google-api-key-here':
        print("❌ Please set your actual Google API key in env.local")
        return
    
    # Create a proper test image (10x10 pixel PNG with some content)
    img = Image.new('RGB', (10, 10), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_data = img_buffer.getvalue()
    test_image_data = base64.b64encode(img_data).decode('utf-8')
    
    # Create test event (simulating API Gateway)
    test_event = {
        'httpMethod': 'POST',
        'body': json.dumps({
            'image': test_image_data,
            'speed': 'fast',
            'imageType': 'unknown'
        })
    }
    
    print(f"📤 Sending test event...")
    print(f"   Image size: {len(test_image_data)} characters (base64)")
    print(f"   Speed: fast")
    
    try:
        # Import and test the Lambda function
        from lambda_function import lambda_handler
        
        print("🔄 Calling lambda_handler...")
        response = lambda_handler(test_event, None)
        
        print(f"\n📥 Response:")
        print(f"   Status Code: {response['statusCode']}")
        
        if response['statusCode'] == 200:
            body = json.loads(response['body'])
            print(f"   Translation: {body.get('translation', 'N/A')}")
            print(f"   Characters: {len(body.get('characters', []))}")
            print(f"   Location: {body.get('location', 'N/A')}")
            print(f"   Processing Time: {body.get('processing_time', 'N/A')}")
            print("\n✅ Test passed! Function is working correctly.")
        else:
            body = json.loads(response['body'])
            print(f"   Error: {body.get('error', 'Unknown error')}")
            print(f"\n❌ Test failed with status code: {response['statusCode']}")
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lambda()