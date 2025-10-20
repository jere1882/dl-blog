import json
import base64
import os
import time
from typing import Dict, Any

from gemini_strategy import analyze_egyptian_art_with_gemini

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for Egyptian art analysis API
    """
    try:
        # Handle CORS preflight requests
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': ''
            }
        
        # Only handle POST requests
        if event.get('httpMethod') != 'POST':
            return {
                'statusCode': 405,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'error': 'Method not allowed. Only POST requests are supported.',
                    'translation': None,
                    'characters': [],
                    'location': None,
                    'processing_time': 'Request failed',
                    'interesting_detail': None,
                    'date': None
                })
            }
        
        # Parse the request body
        body = event.get('body', '')
        if not body:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'error': 'No request body provided',
                    'translation': None,
                    'characters': [],
                    'location': None,
                    'processing_time': 'Request failed',
                    'interesting_detail': None,
                    'date': None
                })
            }
        
        # Parse JSON body
        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'error': 'Invalid JSON in request body',
                    'translation': None,
                    'characters': [],
                    'location': None,
                    'processing_time': 'Request failed',
                    'interesting_detail': None,
                    'date': None
                })
            }
        
        # Extract parameters
        image_data = request_data.get('image')
        speed = request_data.get('speed', 'fast')
        image_type = request_data.get('imageType', 'unknown')
        
        if not image_data:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'error': 'No image data provided in request',
                    'translation': None,
                    'characters': [],
                    'location': None,
                    'processing_time': 'Request failed',
                    'interesting_detail': None,
                    'date': None
                })
            }
        
        # Validate image data (should be base64)
        try:
            # Try to decode to validate it's proper base64
            base64.b64decode(image_data)
        except Exception:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'error': 'Invalid image data. Must be base64 encoded.',
                    'translation': None,
                    'characters': [],
                    'location': None,
                    'processing_time': 'Request failed',
                    'interesting_detail': None,
                    'date': None
                })
            }
        
        print(f"Received image: {len(image_data)} characters (base64)")
        print(f"Analysis settings: speed={speed}, image_type={image_type}")
        
        # Call the Gemini analysis
        print("Calling Gemini API for Egyptian art analysis...")
        gemini_result = analyze_egyptian_art_with_gemini(image_data, speed, image_type)
        
        if gemini_result.get("failure_status") == "success":
            analysis = gemini_result["analysis"]
            response_data = {
                "translation": analysis.get("ancient_text_translation", "No ancient text detected or translation unavailable"),
                "characters": analysis.get("characters", []),
                "location": analysis.get("picture_location", "Location unknown"),
                "processing_time": f"Analysis completed in {gemini_result['api_call_duration']:.2f}s",
                "interesting_detail": analysis.get("interesting_detail", "No notable details identified"),
                "date": analysis.get("date", "Period unknown")
            }
            print("Gemini analysis successful!")
            
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps(response_data, indent=2)
            }
            
        else:
            print(f"Gemini analysis failed: {gemini_result.get('failure_reason', 'Unknown error')}")
            error_details = gemini_result.get('failure_reason', 'Unknown error')
            if 'traceback' in gemini_result:
                error_details += f"\n\nDebug trace:\n{gemini_result['traceback']}"
            
            response_data = {
                "error": error_details,
                "translation": None,
                "characters": [],
                "location": None,
                "processing_time": f"Failed after {gemini_result.get('api_call_duration', 0):.2f}s",
                "interesting_detail": None,
                "date": None
            }
            
            return {
                'statusCode': 500,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps(response_data, indent=2)
            }
            
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        error_details = f"Server error: {str(e)}\n\nDebug trace:\n{traceback.format_exc()}"
        
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                "error": error_details,
                "translation": None,
                "characters": [],
                "location": None,
                "processing_time": "Processing failed",
                "interesting_detail": None,
                "date": None
            }, indent=2)
        }
