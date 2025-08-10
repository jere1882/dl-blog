"use client"

import { useState } from "react"
import {
  AlertCircle,
  CheckCircle,
  FileImage,
  Loader2,
  Upload,
} from "lucide-react"

// UI Components
const Button = ({
  children,
  onClick,
  disabled,
  variant = "default",
  className = "",
}) => {
  const baseClass =
    "px-4 py-2 rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2"
  const variants = {
    default:
      "bg-primary text-primary-foreground hover:bg-primary/90 focus:ring-primary",
    outline:
      "border border-input hover:bg-accent hover:text-accent-foreground focus:ring-accent",
  }
  return (
    <button
      className={`${baseClass} ${variants[variant]} ${
        disabled ? "cursor-not-allowed opacity-50" : ""
      } ${className}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  )
}

const Progress = ({ value, className = "" }) => (
  <div className={`h-2 w-full rounded-full bg-secondary ${className}`}>
    <div
      className="h-2 rounded-full bg-primary transition-all"
      style={{ width: `${value}%` }}
    />
  </div>
)

const Alert = ({ children, variant = "default", className = "" }) => {
  const variants = {
    default: "border-border text-foreground",
    destructive: "border-destructive/50 text-destructive bg-destructive/5",
  }
  return (
    <div className={`rounded-lg border p-4 ${variants[variant]} ${className}`}>
      {children}
    </div>
  )
}

const Card = ({ children, className = "" }) => (
  <div
    className={`rounded-lg border bg-card text-card-foreground shadow-sm ${className}`}
  >
    {children}
  </div>
)

interface Character {
  character_name: string
  reasoning: string
  description: string
  location: string
}

interface AnalysisResult {
  translation?: string
  characters?: Character[]
  location?: string
  error?: string
  interesting_detail?: string
  date?: string
  processing_time?: string
}

interface EgyptianAIAnalyzerProps {
  speed: "regular" | "fast" | "super-fast"
  imageType: "tomb" | "temple" | "other" | "unknown"
}

export function EgyptianAIAnalyzer({
  speed,
  imageType,
}: EgyptianAIAnalyzerProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [state, setState] = useState<
    "idle" | "uploading" | "analyzing" | "completed" | "error"
  >("idle")
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [selectedSample, setSelectedSample] = useState<string | null>(null)

  // Sample images for demo
  const sampleImages = [
    {
      id: "vok1",
      src: "/sample-egyptian-images/VoK.jpg",
      alt: "Egypt vacation photo 1",
      title: "Egypt Photo 1",
    },
    {
      id: "vok2",
      src: "/sample-egyptian-images/VoK2.jpg",
      alt: "Egypt vacation photo 2",
      title: "Egypt Photo 2",
    },
    {
      id: "vok3",
      src: "/sample-egyptian-images/VoK3.jpg",
      alt: "Egypt vacation photo 3",
      title: "Egypt Photo 3",
    },
    {
      id: "vok4",
      src: "/sample-egyptian-images/VoK4.jpg",
      alt: "Egypt vacation photo 4",
      title: "Egypt Photo 4",
    },
    {
      id: "vok5",
      src: "/sample-egyptian-images/VoK5.jpg",
      alt: "Egypt vacation photo 5",
      title: "Egypt Photo 5",
    },
    {
      id: "vok6",
      src: "/sample-egyptian-images/VoK6.jpg",
      alt: "Egypt vacation photo 6",
      title: "Egypt Photo 6",
    },
    {
      id: "vok7",
      src: "/sample-egyptian-images/VoK7.jpg",
      alt: "Egypt vacation photo 7",
      title: "Egypt Photo 7",
    },
    {
      id: "vok8",
      src: "/sample-egyptian-images/VoK8.jpg",
      alt: "Egypt vacation photo 8",
      title: "Egypt Photo 8",
    },
  ]

  const handleSampleSelect = async (sampleSrc: string, sampleId: string) => {
    try {
      // Fetch the sample image
      const response = await fetch(sampleSrc)
      const blob = await response.blob()

      // Create a File object from the blob
      const file = new File([blob], `sample-${sampleId}.jpg`, {
        type: "image/jpeg",
      })

      setSelectedFile(file)
      setPreviewUrl(sampleSrc)
      setSelectedSample(sampleId)
      setState("idle")
      setResult(null)
      setProgress(0)
    } catch (error) {
      console.error("Error loading sample image:", error)
      // Fallback: just set the preview URL for display
      setSelectedFile(null)
      setPreviewUrl(sampleSrc)
      setSelectedSample(sampleId)
      setState("idle")
      setResult(null)
      setProgress(0)
    }
  }

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file)
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
      setSelectedSample(null) // Clear any selected sample
      setState("idle")
      setResult(null)
      setProgress(0)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const analyzeImage = async () => {
    if (!selectedFile && !selectedSample) return

    setState("uploading")
    setProgress(10)

    try {
      let fileToUpload = selectedFile

      // If we have a sample selected but no file (fetch failed), try to fetch it again
      if (selectedSample && !selectedFile && previewUrl) {
        try {
          const response = await fetch(previewUrl)
          const blob = await response.blob()
          fileToUpload = new File([blob], `sample-${selectedSample}.jpg`, {
            type: "image/jpeg",
          })
        } catch (error) {
          console.error("Could not fetch sample image:", error)
          setState("error")
          setResult({
            error:
              "Failed to load sample image. Please try uploading your own image.",
          })
          return
        }
      }

      if (!fileToUpload) return

      // Upload image
      const formData = new FormData()
      formData.append("image", fileToUpload)
      formData.append("speed", speed)
      formData.append("imageType", imageType)

      setProgress(30)
      setState("analyzing")

      // Start progress simulation
      let currentProgress = 30
      let progressCompleted = false

      // Progress simulation: increment 1% per second for up to 60 seconds
      const progressInterval = setInterval(() => {
        if (progressCompleted) {
          clearInterval(progressInterval)
          return
        }

        currentProgress += 1
        if (currentProgress >= 95) {
          currentProgress = 95 // Cap at 95% until API call completes
        }
        setProgress(currentProgress)
      }, 1000) // Update every 1 second (1% per second)

      // Allow switching between local Python server and Next.js API
      const apiBase = process.env.NEXT_PUBLIC_API_BASE || ""
      const endpoint = apiBase
        ? `${apiBase}/analyze`
        : "/api/egyptian-ai-lens/analyze"

      console.log("=== FRONTEND DEBUG START ===")
      console.log("API Base:", apiBase || "(default)")
      console.log("Endpoint:", endpoint)
      console.log(
        "Selected file:",
        selectedFile?.name,
        selectedFile?.size + " bytes"
      )
      console.log("Speed setting:", speed)
      console.log("Image type:", imageType)
      console.log("Environment:", process.env.NODE_ENV)
      console.log("=== MAKING API CALL ===")

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      })

      console.log("=== API RESPONSE RECEIVED ===")
      console.log("Response status:", response.status, response.statusText)
      console.log(
        "Response headers:",
        Object.fromEntries(response.headers.entries())
      )
      console.log("Response ok:", response.ok)

      // Stop progress simulation and complete
      progressCompleted = true
      clearInterval(progressInterval)

      if (!response.ok) {
        const errorText = await response.text()
        console.error("API Error Response:", errorText)
        throw new Error(
          `Analysis failed: ${response.status} ${response.statusText}. Details: ${errorText}`
        )
      }

      const analysisResult = await response.json()
      console.log("=== API SUCCESS ===")
      console.log("Result keys:", Object.keys(analysisResult))
      console.log("Characters found:", analysisResult.characters?.length || 0)
      console.log("Processing time:", analysisResult.processing_time)
      console.log("=== FRONTEND DEBUG END ===")

      setProgress(100)
      setState("completed")
      setResult(analysisResult)
    } catch (error) {
      console.error("=== FRONTEND ERROR ===")
      console.error("Error type:", error?.constructor?.name)
      console.error("Error message:", error?.message)
      console.error("Full error:", error)
      console.error("=== ERROR END ===")

      setState("error")
      setResult({
        error:
          error instanceof Error
            ? `Detailed error: ${error.message}`
            : "An unexpected error occurred",
      })
    }
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <Card className="p-6">
        {!previewUrl ? (
          // Show full upload area when no image selected
          <div
            className="rounded-lg border-2 border-dashed border-border p-8 text-center transition-colors hover:border-border/80 hover:bg-muted/20"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            <div className="space-y-4">
              <FileImage className="mx-auto h-12 w-12 text-muted-foreground" />
              <div>
                <p className="text-lg font-medium">
                  Drop your Egyptian art image here
                </p>
                <p className="text-sm text-muted-foreground">
                  or click to browse
                </p>
              </div>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload">
                <div className="inline-flex cursor-pointer items-center rounded-lg border border-input px-4 py-2 font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2">
                  <Upload className="mr-2 h-4 w-4" />
                  Choose Image
                </div>
              </label>
            </div>
          </div>
        ) : (
          // Show compact area when image is selected
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Selected Image</h3>
              <div className="flex gap-2">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileInput}
                  className="hidden"
                  id="file-change"
                />
                <label htmlFor="file-change">
                  <Button
                    variant="outline"
                    className="cursor-pointer"
                    onClick={() =>
                      document.getElementById("file-change")?.click()
                    }
                    disabled={false}
                  >
                    <Upload className="mr-2 h-4 w-4" />
                    Change Image
                  </Button>
                </label>
              </div>
            </div>

            <img
              src={previewUrl}
              alt="Preview"
              className="mx-auto max-h-96 rounded-lg border object-contain"
            />

            <div className="text-center">
              <Button
                onClick={analyzeImage}
                disabled={state === "uploading" || state === "analyzing"}
                className="px-8"
              >
                {state === "uploading" || state === "analyzing" ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {state === "uploading" ? "Uploading..." : "Analyzing..."}
                  </>
                ) : (
                  "Analyze Egyptian Art"
                )}
              </Button>
            </div>

            {selectedSample && (
              <div className="text-center">
                <p className="text-sm text-muted-foreground">
                  Using sample:{" "}
                  {sampleImages.find((s) => s.id === selectedSample)?.title}
                </p>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Sample Gallery */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="text-center">
            <h3 className="mb-2 text-lg font-semibold">✨ Try Sample Images</h3>
            <p className="text-sm text-muted-foreground">
              Don&apos;t have Egyptian art handy? Try out samples from my 2023
              Egypt vacation
            </p>
          </div>

          {!previewUrl ? (
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4 lg:grid-cols-4">
              {sampleImages.map((sample) => (
                <div
                  key={sample.id}
                  className={`cursor-pointer rounded-lg border-2 transition-all hover:border-primary hover:shadow-md ${
                    selectedSample === sample.id
                      ? "border-primary bg-primary/5"
                      : "border-border"
                  }`}
                  onClick={() => handleSampleSelect(sample.src, sample.id)}
                >
                  <div className="relative aspect-square overflow-hidden rounded-lg">
                    <img
                      src={sample.src}
                      alt={sample.alt}
                      className="h-full w-full object-cover"
                      onError={(e) => {
                        // Fallback for missing images - show a placeholder
                        const target = e.target as HTMLImageElement
                        target.src = `data:image/svg+xml,${encodeURIComponent(`
                          <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
                            <rect width="200" height="200" fill="currentColor" fill-opacity="0.1"/>
                            <text x="100" y="80" text-anchor="middle" fill="currentColor" fill-opacity="0.6" font-size="12">Egyptian Art</text>
                            <text x="100" y="100" text-anchor="middle" fill="currentColor" fill-opacity="0.6" font-size="12">Sample</text>
                            <text x="100" y="120" text-anchor="middle" fill="currentColor" fill-opacity="0.6" font-size="10">${sample.title}</text>
                          </svg>
                        `)}`
                      }}
                    />
                    {selectedSample === sample.id && (
                      <div className="absolute right-2 top-2 rounded-full bg-primary p-1 text-primary-foreground">
                        <CheckCircle className="h-4 w-4" />
                      </div>
                    )}
                  </div>
                  <div className="p-3">
                    <p className="text-center text-sm font-medium">
                      {sample.title}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-3 text-center">
              <p className="text-sm text-muted-foreground">
                Want to try a different sample? Click any image below:
              </p>
              <div className="grid grid-cols-4 gap-2 md:grid-cols-8">
                {sampleImages.map((sample) => (
                  <div
                    key={sample.id}
                    className={`cursor-pointer rounded border-2 transition-all hover:border-primary ${
                      selectedSample === sample.id
                        ? "border-primary bg-primary/5"
                        : "border-border"
                    }`}
                    onClick={() => handleSampleSelect(sample.src, sample.id)}
                  >
                    <div className="relative aspect-square overflow-hidden rounded">
                      <img
                        src={sample.src}
                        alt={sample.alt}
                        className="h-full w-full object-cover"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement
                          target.src = `data:image/svg+xml,${encodeURIComponent(`
                            <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
                              <rect width="100" height="100" fill="currentColor" fill-opacity="0.1"/>
                              <text x="50" y="40" text-anchor="middle" fill="currentColor" fill-opacity="0.6" font-size="8">Egypt</text>
                              <text x="50" y="60" text-anchor="middle" fill="currentColor" fill-opacity="0.6" font-size="8">${sample.id}</text>
                            </svg>
                          `)}`
                        }}
                      />
                      {selectedSample === sample.id && (
                        <div className="absolute right-1 top-1 rounded-full bg-primary p-0.5 text-primary-foreground">
                          <CheckCircle className="h-3 w-3" />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!previewUrl && (
            <div className="pt-2 text-center">
              <p className="text-xs text-muted-foreground">
                Select a sample image above or upload your own Egyptian art
                image
              </p>
            </div>
          )}
        </div>
      </Card>

      {/* Progress */}
      {(state === "uploading" || state === "analyzing") && (
        <Card className="p-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>
                {state === "uploading"
                  ? "Uploading image..."
                  : "AI analyzing Egyptian art..."}
              </span>
              <span>{progress}%</span>
            </div>
            <Progress value={progress} />
          </div>
        </Card>
      )}

      {/* Results */}
      {state === "completed" && result && !result.error && (
        <Card className="p-6">
          <div className="space-y-6">
            <h3 className="flex items-center gap-2 text-lg font-semibold">
              <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
              Analysis Results
            </h3>

            {result.characters && result.characters.length > 0 && (
              <div>
                <h4 className="mb-4 font-semibold">
                  👑 Characters & Figures Identified
                </h4>
                <div className="space-y-4">
                  {result.characters.map((character, index) => (
                    <div
                      key={index}
                      className="rounded-lg border bg-gradient-to-r from-blue-50 to-purple-50 p-4 dark:from-blue-900/20 dark:to-purple-900/20"
                    >
                      <div className="grid gap-3">
                        <div className="flex items-center justify-between">
                          <h5 className="text-lg font-semibold text-primary">
                            {character.character_name}
                          </h5>
                          <span className="rounded-full bg-primary/20 px-3 py-1 text-xs font-medium text-primary">
                            {character.location}
                          </span>
                        </div>

                        <div>
                          <h6 className="mb-1 text-sm font-medium text-muted-foreground">
                            Description:
                          </h6>
                          <p className="text-sm">{character.description}</p>
                        </div>

                        <div>
                          <h6 className="mb-1 text-sm font-medium text-muted-foreground">
                            How was this character identified?
                          </h6>
                          <p className="text-sm italic text-muted-foreground">
                            {character.reasoning}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.translation && (
              <div>
                <h4 className="mb-2 font-semibold">📜 Translation</h4>
                <p className="rounded-lg bg-muted p-4 text-sm">
                  {result.translation}
                </p>
              </div>
            )}

            {result.location && (
              <div>
                <h4 className="mb-2 font-semibold">🗺️ Possible Location</h4>
                <p className="text-sm text-muted-foreground">
                  {result.location}
                </p>
              </div>
            )}

            {result.interesting_detail && (
              <div>
                <h4 className="mb-2 font-semibold">🔍 Interesting Detail</h4>
                <p className="text-sm text-muted-foreground">
                  {result.interesting_detail}
                </p>
              </div>
            )}

            {result.date && (
              <div>
                <h4 className="mb-2 font-semibold">📅 Historical Period</h4>
                <p className="text-sm text-muted-foreground">{result.date}</p>
              </div>
            )}

            {result.processing_time && (
              <div>
                <h4 className="mb-2 font-semibold">⏱️ Processing Time</h4>
                <p className="text-sm text-muted-foreground">
                  {result.processing_time}
                </p>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Error */}
      {(state === "error" || (result && result.error)) && (
        <Alert variant="destructive">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            <span className="font-medium">Analysis Failed</span>
          </div>
          <div className="mt-3 space-y-2">
            <p className="text-sm font-medium">Error Details:</p>
            <div className="rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-sm">
              <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs">
                {result?.error ||
                  "Something went wrong during analysis. Please try again."}
              </pre>
            </div>
            <p className="text-xs text-muted-foreground">
              If this error persists, please check your API configuration or try
              a different image.
            </p>
          </div>
        </Alert>
      )}
    </div>
  )
}
