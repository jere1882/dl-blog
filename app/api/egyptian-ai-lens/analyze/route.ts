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

Your task is to analyze what is depicted in the captured image. Provide a detailed analysis in JSON format with the following structure:

{
  "ancient_text_translation": "Translation of any hieroglyphs or ancient text visible",
  "characters": [
    {
      "character_name": "Name of deity/person",
      "reasoning": "Why you identified them as such",
      "description": "Brief description and facts",
      "location": "Position in image"
    }
  ],
  "picture_location": "Best guess of where this was photographed",
  "interesting_detail": "One fascinating detail an amateur might miss",
  "date": "Historical period when artwork was created"
}

If there are characters depicted (e.g., gods, pharaohs, queens, officials, or other people), identify them by name. For each identified character, provide:
1. **Character Name**: The name of the individual or deity.
2. **Reasoning**: A clear explanation of *why* you identified them as such (e.g., specific regalia, iconography, context).
3. **Description**: Any interesting facts or a brief description of the character/deity.
4. **Location**: Their approximate position in the image (e.g., "far left", "center", "right side", "behind the pharaoh").

For any ancient Egyptian text, hieroglyphs, or symbols, attempt to translate them. If a full translation is not possible due to image quality or complexity, try to identify individual elements, cartouches (especially those containing royal or deity names), or speculate on the general meaning based on context.

Guess the location where the picture was taken (e.g., "Valley of the Kings, Tomb of Tutankhamun (KV62)", "Karnak Temple, Hypostyle Hall"). Be specific if possible, but use speculative language ("possibly", "likely", "could be") if you are not absolutely certain.

Highlight one particularly interesting detail from the image that an amateur might miss but an Egyptologist would find fascinating.

Finally, provide your best guess as to the historical period when the artwork was produced (e.g., "Old Kingdom", "Middle Kingdom", "New Kingdom", "Ptolemaic Period").

IMPORTANT: Respond ONLY with valid JSON. Do not include any other text, explanations, or formatting.`

  if (imageType && imageType !== "unknown") {
    basePrompt += `\n\nHint: The image most likely belongs to a ${imageType}.`
  }

  return basePrompt
}

export async function POST(request: NextRequest) {
  try {
    console.log("Starting Gemini API analysis...")
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

    // Get API key
    const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY
    if (!apiKey) {
      console.error("No Google API key found")
      return NextResponse.json(
        {
          translation: "API configuration error",
          characters: [],
          location: "Unable to process request",
          processing_time: "API key not found",
          interesting_detail:
            "Please configure GOOGLE_API_KEY or GEMINI_API_KEY environment variable",
          date: "Configuration required",
        },
        { status: 500 }
      )
    }

    // Convert image to base64
    const bytes = await image.arrayBuffer()
    const buffer = Buffer.from(bytes)
    const base64Data = buffer.toString("base64")

    console.log(
      `Received image: ${bytes.byteLength} bytes, base64 length: ${base64Data.length}`
    )

    // Initialize Gemini
    const genAI = new GoogleGenerativeAI(apiKey)

    // Select model based on speed
    const modelName =
      speed === "regular"
        ? "gemini-2.5-pro"
        : speed === "super-fast"
        ? "gemini-2.5-flash-lite"
        : "gemini-2.5-flash"

    const model = genAI.getGenerativeModel({
      model: modelName,
      generationConfig: {
        temperature: 0,
        maxOutputTokens: 4096,
      },
    })

    console.log(`Using model: ${modelName}`)

    // Prepare image data
    const imageData = {
      inlineData: {
        data: base64Data,
        mimeType: image.type,
      },
    }

    const prompt = createEgyptianArtPrompt(imageType)
    console.log(`Prompt length: ${prompt.length} characters`)

    // Make API call
    const startTime = Date.now()
    console.log("Calling Gemini API...")

    const result = await model.generateContent([prompt, imageData])
    const response = await result.response
    const text = response.text()

    const duration = (Date.now() - startTime) / 1000
    console.log(`Gemini API call completed in ${duration.toFixed(2)}s`)
    console.log(`Response length: ${text.length} characters`)

    // Parse JSON response
    let analysis: EgyptianArtAnalysis
    try {
      // Clean up the response text
      let cleanedText = text.trim()

      // Remove markdown code blocks if present
      if (cleanedText.startsWith("```json")) {
        cleanedText = cleanedText.slice(7)
      }
      if (cleanedText.endsWith("```")) {
        cleanedText = cleanedText.slice(0, -3)
      }
      cleanedText = cleanedText.trim()

      // Parse JSON
      analysis = JSON.parse(cleanedText)
      console.log("Successfully parsed Gemini response")
    } catch (parseError) {
      console.error("Failed to parse Gemini response as JSON:", parseError)
      console.error("Raw response:", text.substring(0, 500))

      // Fallback response
      return NextResponse.json({
        translation: "Unable to parse hieroglyphs due to response format issue",
        characters: [],
        location: "Egyptian archaeological site (analysis partially failed)",
        processing_time: `Analysis attempted in ${duration.toFixed(2)}s`,
        interesting_detail:
          "The response could not be properly parsed, but the image appears to contain Egyptian artwork",
        date: "Ancient Egyptian period",
      })
    }

    // Return successful analysis
    console.log("Analysis completed successfully")
    return NextResponse.json({
      translation:
        analysis.ancient_text_translation || "No ancient text detected",
      characters: analysis.characters || [],
      location: analysis.picture_location || "Location unknown",
      processing_time: `Analysis completed in ${duration.toFixed(2)}s`,
      interesting_detail:
        analysis.interesting_detail || "No notable details identified",
      date: analysis.date || "Period unknown",
    })
  } catch (error: any) {
    console.error("API route error:", error)
    return NextResponse.json(
      {
        translation: "Analysis failed",
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
