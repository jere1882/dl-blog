#!/usr/bin/env python3
"""
Test the Docker container locally by importing and calling the function directly
"""
import json
import base64
import os
import sys

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, '/var/task')

def test_docker_function():
    """Test the Lambda function inside the Docker container"""
    
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
    
    print("Testing Lambda function inside Docker container...")
    print(f"Event: {json.dumps(test_event, indent=2)}")
    
    try:
        # Import and call the function
        from lambda_function import lambda_handler
        
        response = lambda_handler(test_event, None)
        print(f"\nResponse: {json.dumps(response, indent=2)}")
        
        if response['statusCode'] == 200:
            print("\n✅ Test passed! Function is working correctly in Docker.")
        else:
            print(f"\n❌ Test failed with status code: {response['statusCode']}")
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_docker_function()

