"use client"

import { useState } from "react"

import { EgyptianAIAnalyzer } from "./components/egyptian-ai-analyzer"

export default function EgyptianAILensPage() {
  const [speed, setSpeed] = useState<"regular" | "fast" | "super-fast">("fast")
  const [imageType, setImageType] = useState<
    "tomb" | "temple" | "other" | "unknown"
  >("unknown")

  return (
    <div className="container mx-auto max-w-4xl px-4 py-8">
      <div className="space-y-8">
        {/* Header Section */}
        <div className="space-y-4 text-center">
          <h1 className="text-4xl font-bold tracking-tight">
            🏺 Egyptian AI Lens
          </h1>
          <p className="mx-auto max-w-2xl text-xl text-muted-foreground">
            Unlock the secrets of ancient Egypt! Upload a clear image of
            Egyptian wall paintings, carvings, or tomb art, and discover the
            characters, location, and historical context through AI analysis.
          </p>
        </div>

        {/* What You'll Get */}
        <div className="space-y-3 rounded-lg border bg-card p-6">
          <h2 className="text-lg font-semibold">AI Analysis Includes:</h2>
          <div className="grid gap-4 text-sm md:grid-cols-3">
            <div className="space-y-2">
              <h3 className="font-medium">Deity Identification</h3>
              <p className="text-muted-foreground">
                Identifying gods, goddesses, pharaohs, or notable figures
                depicted
              </p>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium">Location Insights</h3>
              <p className="text-muted-foreground">
                Educated guesses about where the photo might have been taken
              </p>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium">Translation</h3>
              <p className="text-muted-foreground">
                Attempting to interpret ancient Egyptian text and symbols
              </p>
            </div>
          </div>
        </div>

        {/* Analysis Settings & Technical Details - Two Column Layout */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Analysis Settings */}
          <div className="space-y-3 rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold">Analysis Settings</h2>
            <div className="space-y-4">
              {/* Speed Setting */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">
                  Prediction Speed
                </label>
                <div className="flex gap-1">
                  <button
                    type="button"
                    onClick={() => setSpeed("regular")}
                    className={`flex-1 rounded border px-2 py-1 text-xs transition-colors ${
                      speed === "regular"
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-input hover:bg-accent"
                    }`}
                  >
                    Regular
                  </button>
                  <button
                    type="button"
                    onClick={() => setSpeed("fast")}
                    className={`flex-1 rounded border px-2 py-1 text-xs transition-colors ${
                      speed === "fast"
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-input hover:bg-accent"
                    }`}
                  >
                    Fast
                  </button>
                  <button
                    type="button"
                    onClick={() => setSpeed("super-fast")}
                    className={`flex-1 rounded border px-2 py-1 text-xs transition-colors ${
                      speed === "super-fast"
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-input hover:bg-accent"
                    }`}
                  >
                    Super Fast
                  </button>
                </div>
                <p className="text-xs text-muted-foreground">
                  {speed === "super-fast" &&
                    "⚡ Fastest analysis, may affect reliability"}
                  {speed === "fast" &&
                    "🚀 Balanced speed and accuracy (recommended)"}
                  {speed === "regular" &&
                    "🎯 Most thorough analysis, slower processing"}
                </p>
              </div>

              {/* Image Type Setting */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">
                  Image Type Hint
                </label>
                <select
                  value={imageType}
                  onChange={(e) =>
                    setImageType(
                      e.target.value as "tomb" | "temple" | "other" | "unknown"
                    )
                  }
                  className="w-full rounded border border-input bg-background px-2 py-1 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                >
                  <option value="unknown">Unknown / Let AI decide</option>
                  <option value="tomb">
                    A tomb (Valley of Kings, burial sites)
                  </option>
                  <option value="temple">
                    A temple (Karnak, Luxor, Abu Simbel)
                  </option>
                  <option value="other">Other Egyptian art</option>
                </select>
              </div>
            </div>
          </div>

          {/* Technical Details */}
          <div className="space-y-3 rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold">
              🔧 Technical Implementation
            </h2>
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Interested in the technical details? Learn about the
                architecture, Gemini API integration, structured outputs, and
                deployment strategy.
              </p>
              <div className="flex flex-col gap-3">
                <a
                  href="/blog/egyptian-ai-lens-architecture"
                  className="inline-flex items-center justify-center rounded-lg border border-primary/20 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/20"
                >
                  📚 Read Technical Blog Post
                </a>
                <a
                  href="https://github.com/jeremias-rodriguez/dl-blog"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center rounded-lg border border-border bg-muted px-4 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted/80"
                >
                  🔗 View Source Code
                </a>
              </div>
            </div>
          </div>
        </div>

        <EgyptianAIAnalyzer speed={speed} imageType={imageType} />
      </div>
    </div>
  )
}
