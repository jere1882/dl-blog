import { spawn } from "child_process"
import fs from "fs"
import path from "path"
import { NextRequest, NextResponse } from "next/server"

interface TestResult {
  failure_status: string
  failure_reason?: string
  api_call_duration: number
  python_output?: string
  traceback?: string
}

interface GeminiResult {
  failure_status: string
  analysis?: {
    ancient_text_translation: string
    characters: Array<{
      character_name: string
      reasoning: string
      description: string
      location: string
    }>
    picture_location: string
    interesting_detail: string
    date: string
  }
  failure_reason?: string
  api_call_duration: number
  python_output?: string
  python_errors?: string
  traceback?: string
}

async function testPythonModules(): Promise<TestResult> {
  return new Promise((resolve) => {
    const testScript = `
import sys
import traceback

try:
    from pydantic import BaseModel
    from typing import List
    print("✓ Pydantic and typing imports successful")
    
    # Test the schema
    class Character(BaseModel):
        character_name: str
        reasoning: str
        description: str
        location: str

    class EgyptianArtAnalysis(BaseModel):
        ancient_text_translation: str
        characters: List[Character]
        location_guess: str
        interesting_detail: str
        historical_date: str
    
    print("✓ Schema definitions successful")
    
    # Mock response for testing
    mock_result = {
        "failure_status": "success",
        "ancient_text_translation": "Mock hieroglyph translation - Python modules working correctly",
        "characters": [
            {
                "character_name": "Test Deity",
                "reasoning": "Testing character identification system",
                "description": "Mock character for testing purposes",
                "location": "Test location"
            }
        ],
        "location_guess": "Mock location - Egypt",
        "interesting_detail": "This is a test response to verify Python integration",
        "historical_date": "Test period",
        "api_call_duration": 0.1
    }
    
    import json
    print(json.dumps(mock_result))
    
except ImportError as e:
    error_result = {
        "success": False,
        "failure_reason": f"Import error: {str(e)}",
        "traceback": traceback.format_exc(),
        "api_call_duration": 0
    }
    import json
    print(json.dumps(error_result))
except Exception as e:
    error_result = {
        "success": False, 
        "failure_reason": f"General error: {str(e)}",
        "traceback": traceback.format_exc(),
        "api_call_duration": 0
    }
    import json
    print(json.dumps(error_result))
`

    const pythonProcess = spawn("python3", ["-c", testScript])

    let output = ""
    pythonProcess.stdout.on("data", (data: any) => {
      output += data.toString()
    })

    pythonProcess.stderr.on("data", (data: any) => {
      console.error("Python stderr:", data.toString())
    })

    pythonProcess.on("close", () => {
      try {
        const lines = output.trim().split("\n")
        const jsonLine = lines[lines.length - 1]
        const result = JSON.parse(jsonLine)
        resolve(result)
      } catch (error) {
        resolve({
          failure_status: "failure",
          failure_reason: `Failed to parse Python output: ${error}`,
          python_output: output,
          api_call_duration: 0,
        })
      }
    })
  })
}

async function callRealGemini(
  imageBase64: string,
  speed: string,
  imageType: string
): Promise<GeminiResult> {
  return new Promise((resolve) => {
    const tempImagePath = path.join(process.cwd(), "temp_image_data.txt")
    fs.writeFileSync(tempImagePath, imageBase64)

    const realScript = `
import sys
import traceback
import json
import base64
import os

# Load environment variables from .env.local
def load_env_file():
    env_file = '.env.local'
    if os.path.exists(env_file):
        print(f"✓ Found {env_file}", file=sys.stderr)
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✓ Loaded environment variables", file=sys.stderr)
    else:
        print(f"✗ {env_file} not found", file=sys.stderr)

# Load environment variables first
load_env_file()

# Debug: Check if API key is loaded
google_key = os.environ.get('GOOGLE_API_KEY')
gemini_key = os.environ.get('GEMINI_API_KEY')
print(f"Google API Key loaded: {'Yes' if google_key else 'No'}", file=sys.stderr)
print(f"Gemini API Key loaded: {'Yes' if gemini_key else 'No'}", file=sys.stderr)

# Read image data from temp file
with open('temp_image_data.txt', 'r') as f:
    image_b64 = f.read().strip()

try:
    sys.path.append('.')
    from api.egyptian_ai_lens.gemini_strategy import analyze_egyptian_art_with_gemini
    
    print(f"Processing image of {len(image_b64)} characters", file=sys.stderr)
    
    # Call Gemini analysis
    result = analyze_egyptian_art_with_gemini(image_b64, "${speed}", "${imageType}")
    print(json.dumps(result))
    
except Exception as e:
    error_result = {
        "success": False,
        "failure_reason": f"Error in Gemini analysis: {str(e)}",
        "traceback": traceback.format_exc(),
        "api_call_duration": 0
    }
    print(json.dumps(error_result))
`

    const pythonProcess = spawn("python3", ["-c", realScript], {
      cwd: process.cwd(),
    })

    let output = ""
    let errorOutput = ""

    pythonProcess.stdout.on("data", (data: any) => {
      const dataStr = data.toString()
      output += dataStr

      // Forward Python debug output to console for real-time visibility
      console.log("[Python Debug]", dataStr.trim())
    })

    pythonProcess.stderr.on("data", (data: any) => {
      const dataStr = data.toString()
      errorOutput += dataStr

      // Forward Python errors to console for real-time visibility
      console.error("[Python Error]", dataStr.trim())
    })

    pythonProcess.on("close", () => {
      // Clean up temp file
      try {
        fs.unlinkSync(tempImagePath)
      } catch (e) {
        console.warn("Could not clean up temp file:", e)
      }

      try {
        // Try to extract JSON from the last line of output
        const lines = output.trim().split("\n")
        const jsonLine = lines[lines.length - 1]
        const result = JSON.parse(jsonLine)
        resolve(result)
      } catch (error) {
        console.error("Failed to parse Python output:", error)
        console.error("Raw output:", output)
        console.error("Error output:", errorOutput)
        resolve({
          failure_status: "failure",
          failure_reason: `Failed to parse Python response: ${error}`,
          python_output: output,
          python_errors: errorOutput,
          api_call_duration: 0,
        })
      }
    })
  })
}

// Egyptian AI Lens integration for local testing with enhanced debugging!
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    console.log("Starting Python Gemini backend call...")
    const formData = await request.formData()
    const image = formData.get("image") as File
    const speed = (formData.get("speed") as string) || "fast"
    const imageType = (formData.get("imageType") as string) || "unknown"

    if (!image) {
      console.error("No image file provided")
      return NextResponse.json(
        { error: "No image file provided" },
        { status: 400 }
      )
    }

    console.log(`Settings: speed=${speed}, imageType=${imageType}`)

    const bytes = await image.arrayBuffer()
    const buffer = Buffer.from(bytes)
    const imageBase64 = buffer.toString("base64")
    console.log(
      `Received image: ${bytes.byteLength} bytes, base64 length: ${imageBase64.length}`
    )

    // First, test if python works at all
    const pythonTest = spawn("python", ["--version"])

    return new Promise<NextResponse>((resolve) => {
      let pythonVersion = ""
      let pythonError = ""

      pythonTest.stdout.on("data", (data) => {
        pythonVersion += data.toString()
      })

      pythonTest.stderr.on("data", (data) => {
        pythonError += data.toString()
      })

      pythonTest.on("close", (code) => {
        console.log(
          `Python test result: code=${code}, version="${pythonVersion.trim()}", error="${pythonError.trim()}"`
        )

        if (code !== 0) {
          console.error("Python is not available!")
          resolve(
            NextResponse.json(
              {
                translation: "Python environment error",
                characters: [],
                location: "Local development environment",
                processing_time: "Python interpreter not found or not working",
                interesting_detail:
                  "Please ensure Python is installed and accessible",
                date: "Environment check failed",
              },
              { status: 500 }
            )
          )
          return
        }

        console.log("Python available, testing our modules...")
        testPythonModules()
          .then((result) => {
            if (result.failure_status === "success") {
              console.log("Module test successful!")
              callRealGemini(imageBase64, speed, imageType)
                .then((geminiResult) => {
                  if (
                    geminiResult.failure_status === "success" &&
                    geminiResult.analysis
                  ) {
                    const analysis = geminiResult.analysis
                    resolve(
                      NextResponse.json({
                        translation:
                          analysis.ancient_text_translation ||
                          "No ancient text detected",
                        characters: analysis.characters || [],
                        location:
                          analysis.picture_location || "Location unknown",
                        processing_time: `Analysis completed in ${
                          geminiResult.api_call_duration?.toFixed(2) || "N/A"
                        }s`,
                        interesting_detail:
                          analysis.interesting_detail ||
                          "No notable details identified",
                        date: analysis.date || "Period unknown",
                      })
                    )
                    console.log("SUCCESS! Real Gemini analysis completed!")
                  } else {
                    console.log(
                      `Gemini analysis failed: ${geminiResult.failure_reason}`
                    )
                    resolve(
                      NextResponse.json({
                        translation: "Gemini analysis failed",
                        characters: [],
                        location: "Service error",
                        processing_time: `Gemini error: ${geminiResult.failure_reason}`,
                        interesting_detail:
                          geminiResult.traceback || "No additional details",
                        date: "Analysis failed",
                      })
                    )
                  }
                })
                .catch((err) => {
                  console.error("Error calling real Gemini:", err)
                  resolve(
                    NextResponse.json(
                      {
                        translation: "Gemini API call failed",
                        characters: [],
                        location: "Service unavailable",
                      },
                      { status: 500 }
                    )
                  )
                })
            } else {
              console.log(`Module test failed: ${result.failure_reason}`)
              resolve(
                NextResponse.json({
                  translation: "Python module import failed",
                  characters: [],
                  location: "Development environment",
                  processing_time: `Module test failed with code ${result.api_call_duration}`,
                  interesting_detail: result.failure_reason,
                  date: "Import test failed",
                })
              )
            }
          })
          .catch((err) => {
            console.error("Error in module test:", err)
            resolve(
              NextResponse.json(
                {
                  translation: "System error during module test",
                  characters: [],
                  location: "Development environment",
                },
                { status: 500 }
              )
            )
          })
      })
    })
  } catch (error: any) {
    console.error("Top-level error in API route:", error)
    return NextResponse.json(
      {
        translation: "Request processing failed",
        characters: [],
        location: "Unable to process request",
        processing_time: `Error: ${error.message}`,
        interesting_detail: "An error occurred while processing your request",
        date: "Request failed",
      },
      { status: 500 }
    )
  }
}
