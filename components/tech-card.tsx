import Image from "next/image"
import Link from "next/link"
import { FORMAT_H3 } from "@/styles/format"

import { cn } from "@/lib/utils"

const THUMBNAILS = {
  "Computer Vision": "/thumbnails/semantic-segmentation.gif",
  "View Synthesis": "/thumbnails/NERF.png",
  "View Synthesis - Neural Raiance Fields": "/thumbnails/NERF.png",
  "Foundation Models": "/thumbnails/astronomy.jpeg",
  "Egyptian AI Lens": "/sample-egyptian-images/VoK.jpg",
  "foundation-models": "/thumbnails/astronomy.jpeg",
  "computer-vision": "/thumbnails/semantic-segmentation.gif",
  "view-synthesis": "/thumbnails/NERF.png",
  "egyptian-ai-lens": "/sample-egyptian-images/VoK.jpg",
}

export function TechCard({
  name,
  description,
  link,
}: {
  name: string
  description: string
  link: string
}) {
  // Get the thumbnail, fallback to semantic segmentation if not found
  const thumbnailSrc =
    THUMBNAILS[name] || "/thumbnails/semantic-segmentation.gif"

  return (
    <Link href={link}>
      <div className="relative animate-border overflow-hidden rounded-lg bg-gradient-to-r from-border via-primary to-border bg-[length:400%_400%] p-px ">
        <span className="block rounded-md bg-background hover:bg-accent">
          <div className="flex h-72 flex-col justify-between rounded-md p-4">
            <div className="flex flex-1 flex-col space-y-4">
              <div className="flex items-center gap-3">
                <h3 className={cn(FORMAT_H3, "mt-0 text-lg font-semibold")}>
                  {name}
                </h3>
              </div>
              <p className="line-clamp-3 flex-1 text-sm text-muted-foreground">
                {description}
              </p>
              <div className="flex items-end justify-center">
                <div className="relative h-28 w-36 overflow-hidden rounded-lg border border-border/20">
                  <Image
                    src={thumbnailSrc}
                    alt={`${name} illustration`}
                    fill
                    className="object-cover"
                    sizes="(max-width: 144px) 100vw, 144px"
                  />
                </div>
              </div>
            </div>
          </div>
        </span>
      </div>
    </Link>
  )
}
