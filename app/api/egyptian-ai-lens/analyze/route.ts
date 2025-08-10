import { NextRequest, NextResponse } from "next/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

interface Character {
  character_name: string
  reasoning: string
  description: string
  location: string
}

interface EgyptianArtAnalysis {
  ancient_text_translation: string
  characters: Character[]
  picture_location: string
  interesting_detail: string
  date: string
}

function createEgyptianArtPrompt(imageType?: string): string {
  let basePrompt = `You are an expert Egyptologist with deep knowledge of ancient Egyptian art, tomb paintings, temple reliefs, and ancient texts. You are analyzing a photograph taken by a tourist of ancient Egyptian wall decorations, likely from famous sites like the Valley of the Kings, Karnak Temple, or other well-documented locations.

**IMPORTANT: Use your extensive knowledge of famous Egyptian tombs and their documented artwork, especially:**
- Tutankhamun's tomb (KV62) and its famous painted scenes
- Other Valley of the Kings tombs (KV1-KV64)
- Well-documented temple reliefs from Karnak, Luxor, Abu Simbel
- Famous Egyptian artworks

Your task is to analyze what is depicted in the captured image. Provide a detailed analysis in the specified JSON format.

If there are characters depicted (e.g., gods, pharaohs, queens, officials, or other people), identify them by name. For each identified character, provide:
1. **Character Name**: The name of the individual or deity.
2. **Reasoning**: A clear explanation of *why* you identified them as such (e.g., specific regalia, iconography, context).
3. **Description**: Any interesting facts or a brief description of the character/deity.
4. **Location**: Their approximate position in the image (e.g., "far left", "center", "right side", "behind the pharaoh").

For any ancient Egyptian text, hieroglyphs, or symbols, attempt to translate them. If a full translation is not possible due to image quality or complexity, try to identify individual elements, cartouches (especially those containing royal or deity names), or speculate on the general meaning based on context.

Guess the location where the picture was taken (e.g., "Valley of the Kings, Tomb of Tutankhamun (KV62)", "Karnak Temple, Hypostyle Hall"). Be specific if possible, but use speculative language ("possibly", "likely", "could be") if you are not absolutely certain.

Highlight one particularly interesting detail from the image that an amateur might miss but an Egyptologist would find fascinating.

Finally, provide your best guess as to the historical period when the artwork was produced (e.g., "Old Kingdom", "Middle Kingdom", "New Kingdom", "Ptolemaic Period").

Return your analysis as a JSON object with this exact structure:
{
  "ancient_text_translation": "Translation of any hieroglyphs or text",
  "characters": [
    {
      "character_name": "Name of character/deity",
      "reasoning": "Why you identified them this way",
      "description": "Interesting facts about this character",
      "location": "Position in image"
    }
  ],
  "picture_location": "Likely location where photo was taken",
  "interesting_detail": "Expert insight about the artwork",
  "date": "Historical period estimate"
}`

  if (imageType && imageType !== "unknown") {
    basePrompt += `\n\nHint: The image most likely belongs to a ${imageType}.`
  }

  return basePrompt
}

// Egyptian AI Lens integration for production with enhanced debugging
export async function POST(request: NextRequest) {
  const requestId = Math.random().toString(36).substring(7)

  console.log(`[${requestId}] === EGYPTIAN AI LENS API ROUTE START ===`)
  console.log(`[${requestId}] Environment check:`)
  console.log(`[${requestId}] - NODE_ENV: ${process.env.NODE_ENV}`)
  console.log(`[${requestId}] - VERCEL: ${process.env.VERCEL}`)
  console.log(
    `[${requestId}] - GOOGLE_API_KEY exists: ${!!process.env.GOOGLE_API_KEY}`
  )
  console.log(
    `[${requestId}] - GEMINI_API_KEY exists: ${!!process.env.GEMINI_API_KEY}`
  )

  try {
    console.log(`[${requestId}] Parsing form data...`)
    const formData = await request.formData()
    const image = formData.get("image") as File
    const speed = (formData.get("speed") as string) || "fast"
    const imageType = (formData.get("imageType") as string) || "unknown"

    console.log(`[${requestId}] Request parameters:`)
    console.log(`[${requestId}] - speed: ${speed}`)
    console.log(`[${requestId}] - imageType: ${imageType}`)
    console.log(
      `[${requestId}] - image: ${
        image ? `${image.name} (${image.size} bytes)` : "null"
      }`
    )

    if (!image) {
      console.error(`[${requestId}] ERROR: No image file provided`)
      return NextResponse.json(
        {
          translation: "Analysis failed",
          characters: [],
          location: "No image provided",
          processing_time: "Error: No image file provided",
          interesting_detail: "Error: No image file was uploaded",
          date: "Request failed",
        },
        { status: 400 }
      )
    }

    // Check for API key
    const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY
    if (!apiKey) {
      console.error(`[${requestId}] ERROR: No API key found`)
      return NextResponse.json(
        {
          translation: "Analysis failed",
          characters: [],
          location: "API configuration error",
          processing_time: "Error: No API key configured",
          interesting_detail:
            "Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set in Vercel deployment",
          date: "Request failed",
        },
        { status: 500 }
      )
    }

    console.log(`[${requestId}] Converting image to base64...`)
    const bytes = await image.arrayBuffer()
    const buffer = Buffer.from(bytes)

    console.log(`[${requestId}] Image processing:`)
    console.log(`[${requestId}] - Buffer size: ${buffer.length} bytes`)
    console.log(`[${requestId}] - Image type: ${image.type}`)

    // Initialize Google AI
    console.log(`[${requestId}] Initializing Google Generative AI...`)
    const genAI = new GoogleGenerativeAI(apiKey)

    // Select model based on speed
    const modelMap = {
      regular: "gemini-1.5-pro",
      fast: "gemini-1.5-flash",
      "super-fast": "gemini-1.5-flash",
    }
    const modelName =
      modelMap[speed as keyof typeof modelMap] || "gemini-1.5-flash"

    console.log(`[${requestId}] Using model: ${modelName}`)

    const model = genAI.getGenerativeModel({ model: modelName })

    // Create prompt
    const prompt = createEgyptianArtPrompt(imageType)
    console.log(`[${requestId}] Prompt length: ${prompt.length} characters`)

    // Prepare image data
    const imagePart = {
      inlineData: {
        data: buffer.toString("base64"),
        mimeType: image.type || "image/jpeg",
      },
    }

    console.log(`[${requestId}] Making API call to ${modelName}...`)
    const startTime = Date.now()

    const result = await model.generateContent([prompt, imagePart])

    const duration = Date.now() - startTime
    console.log(`[${requestId}] API call completed in ${duration}ms`)

    const response = await result.response
    const text = response.text()

    console.log(`[${requestId}] Response received:`)
    console.log(`[${requestId}] - Response length: ${text.length} characters`)
    console.log(`[${requestId}] - First 200 chars: ${text.substring(0, 200)}`)

    // Parse JSON response
    let analysisData: EgyptianArtAnalysis
    try {
      analysisData = JSON.parse(text)
      console.log(`[${requestId}] JSON parsing successful`)
    } catch (parseError) {
      console.error(`[${requestId}] JSON parsing failed:`, parseError)
      console.error(`[${requestId}] Raw response: ${text}`)

      return NextResponse.json(
        {
          translation: "Analysis failed",
          characters: [],
          location: "Response parsing error",
          processing_time: `Error: Failed to parse AI response after ${duration}ms`,
          interesting_detail: `Error: The AI returned malformed JSON. Raw response: ${text.substring(
            0,
            500
          )}...`,
          date: "Request failed",
        },
        { status: 500 }
      )
    }

    console.log(`[${requestId}] Analysis complete:`)
    console.log(
      `[${requestId}] - Characters found: ${
        analysisData.characters?.length || 0
      }`
    )
    console.log(
      `[${requestId}] - Location: ${analysisData.picture_location?.substring(
        0,
        50
      )}...`
    )

    // Format the response to match the expected frontend format
    const formattedResponse = {
      translation:
        analysisData.ancient_text_translation || "No ancient text detected",
      characters: analysisData.characters || [],
      location: analysisData.picture_location || "Location unknown",
      processing_time: `${duration}ms`,
      interesting_detail:
        analysisData.interesting_detail || "No specific details identified",
      date: analysisData.date || "Period unknown",
    }

    console.log(`[${requestId}] === REQUEST COMPLETED SUCCESSFULLY ===`)
    return NextResponse.json(formattedResponse)
  } catch (error: any) {
    console.error(`[${requestId}] === ERROR OCCURRED ===`)
    console.error(`[${requestId}] Error type: ${error.constructor.name}`)
    console.error(`[${requestId}] Error message: ${error.message}`)
    console.error(`[${requestId}] Error stack:`, error.stack)

    // Check for specific error types
    let errorDetail = error.message
    if (error.message?.includes("API_KEY")) {
      errorDetail = "API key is invalid or missing"
    } else if (error.message?.includes("quota")) {
      errorDetail = "API quota exceeded"
    } else if (error.message?.includes("rate limit")) {
      errorDetail = "Rate limit exceeded"
    } else if (error.message?.includes("fetch")) {
      errorDetail = "Network connection error"
    }

    return NextResponse.json(
      {
        translation: "Analysis failed",
        characters: [],
        location: "Error occurred",
        processing_time: `Error after ${Date.now()}ms`,
        interesting_detail: `Detailed error: ${errorDetail}. Check Vercel function logs for more details.`,
        date: "Request failed",
      },
      { status: 500 }
    )
  }
}
