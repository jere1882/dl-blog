import { siteConfig } from "@/config/site"
import { Icons } from "@/components/icons"

export function SiteFooter() {
  return (
    <footer className="container">
      <div className="flex  flex-col items-center justify-between gap-4 border-t border-t-secondary py-10 md:h-24 md:flex-row md:py-0">
        <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
          <Icons.logo />
          <p className="text-center text-sm leading-loose md:text-left">
            Built by{" "}
            <a
              href={siteConfig.links.linkedin}
              target="_blank"
              rel="noreferrer"
              className="font-medium underline underline-offset-4"
            >
              Jeremias Rodriguez
            </a>
            . Based on Francisco Moretti&apos;s
            <a
              href={siteConfig.links.fran_github}
              target="_blank"
              rel="noreferrer"
              className="font-medium underline underline-offset-4"
            >
              Site
            </a>
          </p>
        </div>
      </div>
    </footer>
  )
}
