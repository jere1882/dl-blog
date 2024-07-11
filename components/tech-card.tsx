import Image from "next/image"
import Link from "next/link"
import basicsIcon from "@/assets/logos/basics.png"
import ARresearchIcon from "@/assets/logos/conicet.jpg"
import interpretabilityIcon from "@/assets/logos/interpretability.jpeg"
import computerVisionIcon from "@/assets/logos/semseg.gif"
import transformerIcon from "@/assets/logos/transformers.png"
import { FORMAT_H3 } from "@/styles/format"

import { cn } from "@/lib/utils"

const ICONS = {
  "Computer Vision": computerVisionIcon,
  Transformers: transformerIcon,
  "Deep Learning Basics": basicsIcon,
  Interpretability: interpretabilityIcon,
  "Research in Argentina": ARresearchIcon,
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
  return (
    <Link href={link}>
      <div className="relative animate-border overflow-hidden rounded-lg bg-gradient-to-r from-border via-primary to-border bg-[length:400%_400%] p-px ">
        <span className="block rounded-md bg-background hover:bg-accent">
          <div className="h-65 flex flex-col justify-between rounded-md p-4">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <h3 className={cn(FORMAT_H3, "mt-0")}>{name}</h3>
              </div>
              <p className="line-clamp-4 text-sm text-muted-foreground">
                {description}
              </p>
              <div className="flex justify-center">
                <Image
                  src={ICONS[name]}
                  alt={`${name} icon`}
                  className="w-45 h-40"
                />
              </div>
            </div>
          </div>
        </span>
      </div>
    </Link>
  )
}
