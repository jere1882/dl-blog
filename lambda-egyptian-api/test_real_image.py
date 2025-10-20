#!/usr/bin/env python3
import base64
import json
import requests

# Read the image file
image_path = "../public/sample-egyptian-images/VoK.jpg"
with open(image_path, "rb") as image_file:
    image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

# API endpoint and key
api_url = "https://agmlb96l0m.execute-api.us-east-2.amazonaws.com/default/egyptianArtAnalyzer"
api_key = "AI90LZxG6LJdK8ESosT93nwn27FlNgj7jjNPiLai"

# Prepare the request
payload = {
    "image": image_base64,
    "speed": "fast",
    "imageType": "image/jpeg"
}

headers = {
    "Content-Type": "application/json",
    "x-api-key": api_key
}

print("Sending Egyptian image to Lambda API...")
print(f"Image: {image_path}")
print(f"API: {api_url}")

try:
    response = requests.post(api_url, json=payload, headers=headers, timeout=60)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n=== ANALYSIS RESULT ===")
        print(f"Translation: {result.get('translation', 'N/A')}")
        print(f"Characters: {result.get('characters', [])}")
        print(f"Location: {result.get('location', 'N/A')}")
        print(f"Interesting Detail: {result.get('interesting_detail', 'N/A')}")
        print(f"Processing Time: {result.get('processing_time', 'N/A')}")
    else:
        print(f"Error: {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except Exception as e:
    print(f"Error: {e}")