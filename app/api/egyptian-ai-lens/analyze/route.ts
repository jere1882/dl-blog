// Deprecated: Node route disabled in favor of Python Vercel Function at /api/egyptian_ai_lens/analyze
import { NextResponse } from "next/server"

// Deprecated: Node route disabled in favor of Python Vercel Function at /api/egyptian_ai_lens/analyze

export const runtime = "nodejs"
export const dynamic = "force-dynamic"

export async function POST() {
  return NextResponse.json(
    {
      error: "Deprecated. Use Python function: /api/egyptian_ai_lens/analyze",
    },
    { status: 410 }
  )
}
