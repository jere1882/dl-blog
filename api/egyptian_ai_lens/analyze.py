import json
import base64
import io
from http.server import BaseHTTPRequestHandler
import cgi
from .gemini_strategy import analyze_egyptian_art_with_gemini

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Set CORS headers
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Parse multipart form data manually
            content_type = self.headers.get('Content-Type', '')
            if not content_type.startswith('multipart/form-data'):
                self._send_error_response("Content-Type must be multipart/form-data")
                return
                
            boundary = None
            for part in content_type.split(';'):
                part = part.strip()
                if part.startswith('boundary='):
                    boundary = part[9:]
                    break
            
            if not boundary:
                self._send_error_response("No boundary found in Content-Type")
                return
                
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error_response("No content provided")
                return
                
            raw_data = self.rfile.read(content_length)
            
            boundary_bytes = f'--{boundary}'.encode()
            parts = raw_data.split(boundary_bytes)
            
            image_data = None
            speed = 'fast'  # default
            image_type = 'unknown'  # default
            
            for part in parts:
                if b'name="image"' in part and b'Content-Type:' in part:
                    data_start = part.find(b'\r\n\r\n')
                    if data_start != -1:
                        image_data = part[data_start + 4:]
                        if image_data.endswith(b'\r\n'):
                            image_data = image_data[:-2]
                elif b'name="speed"' in part:
                    data_start = part.find(b'\r\n\r\n')
                    if data_start != -1:
                        speed_data = part[data_start + 4:]
                        if speed_data.endswith(b'\r\n'):
                            speed_data = speed_data[:-2]
                        speed = speed_data.decode('utf-8')
                elif b'name="imageType"' in part:
                    data_start = part.find(b'\r\n\r\n')
                    if data_start != -1:
                        image_type_data = part[data_start + 4:]
                        if image_type_data.endswith(b'\r\n'):
                            image_type_data = image_type_data[:-2]
                        image_type = image_type_data.decode('utf-8')
            
            if not image_data:
                self._send_error_response("No image data found in request")
                return
            
            # Convert to base64 for Gemini API
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            print(f"Received image: {len(image_data)} bytes")
            print(f"Analysis settings: speed={speed}, image_type={image_type}")
            
            print("Calling Gemini API for Egyptian art analysis...")
            gemini_result = analyze_egyptian_art_with_gemini(image_base64, speed, image_type)
            
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
                
            else:
                print(f"Gemini analysis failed: {gemini_result.get('failure_reason', 'Unknown error')}")
                # Include debug information in the error
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
                        
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            import traceback
            error_details = f"Server error: {str(e)}\n\nDebug trace:\n{traceback.format_exc()}"
            
            response_data = {
                "error": error_details,
                "translation": None,
                "characters": [],
                "location": None,
                "processing_time": "Processing failed",
                "interesting_detail": None,
                "date": None
            }
        
        try:
            # Send JSON response
            response_json = json.dumps(response_data, indent=2)
            self.wfile.write(response_json.encode('utf-8'))
            print("Response sent successfully!")
            
        except Exception as e:
            print(f"Error sending response: {str(e)}")
    
    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        print("CORS preflight handled")
    
    def _send_error_response(self, message: str):
        """Helper method to send error responses"""
        try:
            self.send_response(400)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            error_response = {
                "error": f"Request processing failed: {message}",
                "translation": None,
                "characters": [],
                "location": None,
                "processing_time": "Request failed",
                "interesting_detail": None,
                "date": None
            }
            
            response_json = json.dumps(error_response, indent=2)
            self.wfile.write(response_json.encode('utf-8'))
            print(f"Error response sent: {message}")
            
        except Exception as e:
            print(f"Error sending error response: {str(e)}") 