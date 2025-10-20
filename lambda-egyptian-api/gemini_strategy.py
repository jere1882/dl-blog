import os
import time
import base64
import json
import google.generativeai as genai
import io
import PIL.Image
from schemas import EgyptianArtAnalysis
import sys

DEFAULT_THINKING_BUDGET = 2000

def create_egyptian_art_prompt(image_type_hint=None):
    """Create the expert Egyptologist prompt for analyzing Egyptian art images."""
    base_prompt = """You are an expert Egyptologist with deep knowledge of ancient Egyptian art, tomb paintings, temple reliefs, and ancient texts. You are analyzing a photograph taken by a tourist of ancient Egyptian wall decorations, likely from famous sites like the Valley of the Kings, Karnak Temple, or other well-documented locations.

**IMPORTANT: Use your extensive knowledge of famous Egyptian tombs and their documented artwork, especially:**
- Tutankhamun's tomb (KV62) and its famous painted scenes
- Other Valley of the Kings tombs (KV1-KV64)
- Well-documented temple reliefs from Karnak, Luxor, Abu Simbel
- Famous Egyptian artworks

Your task is to analyze what is depicted in the captured image. Provide a detailed analysis in the specified JSON format.

If there are characters depicted (e.g., gods, pharaohs, queens, officials, or other people), identify them by name. For each identified character, provide:
1.  **Character Name**: The name of the individual or deity.
2.  **Reasoning**: A clear explanation of *why* you identified them as such (e.g., specific regalia, iconography, context).
3.  **Description**: Any interesting facts or a brief description of the character/deity.
4.  **Location**: Their approximate position in the image (e.g., "far left", "center", "right side", "behind the pharaoh").

For any ancient Egyptian text, hieroglyphs, or symbols, attempt to translate them. If a full translation is not possible due to image quality or complexity, try to identify individual elements, cartouches (especially those containing royal or deity names), or speculate on the general meaning based on context.

Guess the location where the picture was taken (e.g., "Valley of the Kings, Tomb of Tutankhamun (KV62)", "Karnak Temple, Hypostyle Hall"). Be specific if possible, but use speculative language ("possibly", "likely", "could be") if you are not absolutely certain.

Highlight one particularly interesting detail from the image that an amateur might miss but an Egyptologist would find fascinating.

Finally, provide your best guess as to the historical period when the artwork was produced (e.g., "Old Kingdom", "Middle Kingdom", "New Kingdom", "Ptolemaic Period")."""
    
    if image_type_hint and image_type_hint != 'unknown':
        base_prompt += f"\n\nHint: The image most likely belongs to a {image_type_hint}."
    
    return base_prompt

def analyze_egyptian_art_with_gemini(image_data, speed='fast', image_type='unknown', thinking_budget=DEFAULT_THINKING_BUDGET):
    """
    Analyze Egyptian art image using Gemini with structured output.
    
    Args:
        image_data: Base64-encoded image data
        speed: 'regular' (gemini-2.5-pro), 'fast' (gemini-2.5-flash), 'super-fast' (gemini-2.5-flash-lite)
        image_type: 'tomb', 'temple', 'other', or 'unknown'
        thinking_budget: Thinking budget for the model
    
    Returns:
        Dict containing analysis results or error information
    """
    try:
        api_call_duration = 0  # Initialize in case of early errors
        api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return {
                "failure_status": "api_failure",
                "failure_reason": "No Google API key found in environment variables",
                "api_call_duration": 0
            }

        genai.configure(api_key=api_key)
        
        image_bytes = base64.b64decode(image_data)
        image = PIL.Image.open(io.BytesIO(image_bytes))
        
        prompt_text = create_egyptian_art_prompt(image_type)
        
        # Select model based on speed setting
        speed_to_model = {
            'regular': 'gemini-2.5-pro',
            'fast': 'gemini-2.5-flash', 
            'super-fast': 'gemini-2.5-flash-lite'
        }
        model_name = speed_to_model.get(speed, 'gemini-2.5-flash')
        
        # Initialize retry configuration before debug logging
        max_retries = 2
        
        print(f"=== GEMINI API CALL DEBUG INFO ===")
        print(f"Model: {model_name}")
        print(f"Speed setting: {speed}")
        print(f"Image type hint: {image_type}")
        print(f"Thinking budget: {thinking_budget}")
        print(f"Image data size: {len(image_data)} characters (base64)")
        print(f"Prompt length: {len(prompt_text)} characters")
        print(f"Max retries: {max_retries}")
        print(f"Temperature: 0 (deterministic)")
        print(f"Response format: JSON with structured schema")
        print(f"Schema: EgyptianArtAnalysis (characters, picture_location, interesting_detail, date, ancient_text_translation)")
        print(f"=== END DEBUG INFO ===")
        
        api_call_start_time = time.time()
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    # Exponential backoff: wait 1s, then 2s, then 4s, etc.
                    wait_time = 2 ** (retry_count - 1)
                    print(f"RETRY: Gemini call retry #{retry_count + 1}/{max_retries + 1} after {wait_time}s wait")
                    time.sleep(wait_time)
                else:
                    print(f"STARTING: Initial Gemini API call to {model_name}")
                
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=genai.types.GenerationConfig(
                        response_schema=EgyptianArtAnalysis,
                        response_mime_type="application/json",
                        temperature=0
                    )
                )
                
                print(f"MODEL CREATED: {model_name} with JSON schema enforcement")
                print(f"CALLING: generate_content() with prompt + image...")
                
                response = model.generate_content([prompt_text, image])
                
                # If we get here, the call succeeded
                print(f"SUCCESS: API call completed in {time.time() - api_call_start_time:.2f}s")
                print(f"RESPONSE TYPE: {type(response)}")
                print(f"RESPONSE CANDIDATES: {len(response.candidates) if hasattr(response, 'candidates') else 'N/A'}")
                break
                
            except Exception as e:
                error_str = str(e).lower()
                is_5xx_error = any(code in error_str for code in ['500', '502', '503', '504', '429', 'rate limit', 'quota', 'internal error', 'service unavailable', 'timeout'])
                
                if is_5xx_error and retry_count < max_retries:
                    retry_count += 1
                    print(f"Gemini API error (5xx): {str(e)}. Retrying... ({retry_count}/{max_retries})")
                    continue
                else:
                    # Either not a 5xx error, or we've exhausted retries
                    raise e
        
        api_call_duration = time.time() - api_call_start_time
        
        # Try different ways to access the response text
        try:
            # Method 1: Direct text access
            response_text = response.text
        except AttributeError:
            try:
                # Method 2: Through candidates
                response_text = response.candidates[0].content.parts[0].text
            except AttributeError:
                try:
                    # Method 3: Check if parts contain text differently
                    part = response.candidates[0].content.parts[0]
                    response_text = getattr(part, 'text', str(part))
                except Exception as e:
                    raise Exception(f"Cannot access response text. Response structure: {type(response)}")
        
        # Debug: Log the raw response for troubleshooting
        print(f"Raw Gemini response (first 500 chars): {response_text[:500]}")
        print(f"Raw Gemini response (last 200 chars): {response_text[-200:]}")
        print(f"Response length: {len(response_text)} characters")
        
        # Try to parse JSON with better error handling
        try:
            analysis_data = json.loads(response_text)
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing failed: {json_error}")
            print(f"Error at position {json_error.pos}: '{response_text[max(0, json_error.pos-50):json_error.pos+50]}'")
            
            # Try to fix common JSON issues
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            # Try parsing the cleaned text
            try:
                analysis_data = json.loads(cleaned_text)
                print("Successfully parsed after cleaning")
            except json.JSONDecodeError:
                # Last resort: try to extract just the JSON part
                import re
                json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if json_match:
                    try:
                        analysis_data = json.loads(json_match.group())
                        print("Successfully extracted and parsed JSON")
                    except json.JSONDecodeError:
                        raise Exception(f"Could not parse Gemini response as JSON. Raw response: {response_text[:1000]}...")
                else:
                    raise Exception(f"No JSON found in Gemini response. Raw response: {response_text[:1000]}...")
        
        analysis = EgyptianArtAnalysis(**analysis_data)
        
        # Final debug summary
        print(f"=== FINAL RESULT SUMMARY ===")
        print(f"Status: SUCCESS")
        print(f"Total duration: {api_call_duration:.2f}s")
        print(f"Characters found: {len(analysis.characters)}")
        print(f"Location: {analysis.picture_location[:50]}{'...' if len(analysis.picture_location) > 50 else ''}")
        print(f"Historical period: {analysis.date}")
        print(f"Translation length: {len(analysis.ancient_text_translation)} chars")
        print(f"=== END SUMMARY ===")
        
        return {
            "failure_status": "success",
            "analysis": analysis.model_dump(),
            "api_call_duration": api_call_duration,
            "raw_response": analysis_data
        }
            
    except Exception as e:
        import traceback
        api_call_duration = time.time() - api_call_start_time if 'api_call_start_time' in locals() else 0
        print(f"=== ERROR SUMMARY ===")
        print(f"Status: FAILED")
        print(f"Duration before failure: {api_call_duration:.2f}s")
        print(f"Error: {str(e)}")
        print(f"=== END ERROR SUMMARY ===")
        return {
            "failure_status": "api_failure",
            "failure_reason": f"Gemini API call failed: {str(e)}",
            "api_call_duration": api_call_duration,
            "traceback": traceback.format_exc()
        } 