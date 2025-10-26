import Image from "next/image"
import Link from "next/link"
import { FORMAT_H3 } from "@/styles/format"

import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"

const THUMBNAILS = {
  "Computer Vision": "/thumbnails/semantic-segmentation.gif",
  "View Synthesis": "/thumbnails/NERF.png",
  "View Synthesis - Neural Raiance Fields": "/thumbnails/NERF.png",
  "Foundation Models": "/thumbnails/astronomy.jpeg",
  "Egyptian AI Lens": "/sample-egyptian-images/VoK.jpg",
  "Document Hyper Resolution": "/thumbnails/blocks.jpeg",
  "foundation-models": "/thumbnails/astronomy.jpeg",
  "computer-vision": "/thumbnails/semantic-segmentation.gif",
  "view-synthesis": "/thumbnails/NERF.png",
  "egyptian-ai-lens": "/sample-egyptian-images/VoK.jpg",
  "document-hyper-resolution": "/thumbnails/blocks.jpeg",
  // Add exact matches for the new card names
  "Semantic Segmentation of Underwater Scenery":
    "/thumbnails/semantic-segmentation.gif",
  "View Synthesis: Indoor Scene Reconstruction": "/thumbnails/NERF.png",
  "Foundation Models for Astronomy": "/thumbnails/astronomy.jpeg",
}

export function TechCard({
  name,
  description,
  link,
  tags = [],
}: {
  name: string
  description: string
  link: string
  tags?: string[]
}) {
  // Get the thumbnail, fallback to semantic segmentation if not found
  const thumbnailSrc =
    THUMBNAILS[name] || "/thumbnails/semantic-segmentation.gif"

  return (
    <Link href={link}>
      <div className="relative animate-border overflow-hidden rounded-lg bg-gradient-to-r from-border via-primary to-border bg-[length:400%_400%] p-px ">
        <span className="block rounded-md bg-background hover:bg-accent">
          <div className="flex h-[380px] flex-col rounded-md p-5">
            {/* Title - fixed height */}
            <div className="mb-3 h-10">
              <h3
                className={cn(
                  FORMAT_H3,
                  "mt-0 line-clamp-2 text-sm font-bold uppercase leading-tight tracking-wide"
                )}
              >
                {name}
              </h3>
            </div>

            {/* Description - fixed height, no clamp to avoid ... */}
            <div className="mb-4 h-14">
              <p className="text-xs leading-snug text-muted-foreground">
                {description}
              </p>
            </div>

            {/* Small spacer */}
            <div className="h-2" />

            {/* Image - fixed position from bottom */}
            <div className="relative mb-3 h-36 w-full overflow-hidden rounded-lg border border-border/20">
              <Image
                src={thumbnailSrc}
                alt={`${name} illustration`}
                fill
                className="object-cover"
                sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 25vw"
              />
            </div>

            {/* Tags - fixed height */}
            <div className="flex h-7 flex-wrap gap-1.5">
              {tags.map((tag) => (
                <Badge
                  key={tag}
                  variant="secondary"
                  className="h-fit py-0.5 text-xs"
                >
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        </span>
      </div>
    </Link>
  )
}
