"use client"

import * as React from "react"

import { TableOfContents } from "@/lib/toc"
import { cn } from "@/lib/utils"
import { useMounted } from "@/hooks/use-mounted"

interface BlogTocProps {
  toc: TableOfContents
}

export function BlogTableOfContents({ toc }: BlogTocProps) {
  const itemIds = React.useMemo(
    () =>
      toc.items
        ? toc.items
            .flatMap((item) => [item.url, item?.items?.map((item) => item.url)])
            .flat()
            .filter(Boolean)
            .map((id) => id?.split("#")[1])
        : [],
    [toc]
  )
  const activeHeading = useActiveItem(itemIds)
  const mounted = useMounted()

  if (!toc?.items?.length) {
    return null
  }

  if (!mounted) {
    return null
  }

  return (
    <div className="space-y-2 border-l-2 pl-6">
      <p className="mb-4 text-lg font-semibold">Table of Contents</p>
      <Tree tree={toc} activeItem={activeHeading} />
    </div>
  )
}

function useActiveItem(itemIds: (string | undefined)[]) {
  const [activeId, setActiveId] = React.useState<string>("")

  React.useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id)
          }
        })
      },
      { rootMargin: `-10% 0% -75% 0%` }
    )

    itemIds?.forEach((id) => {
      if (!id) {
        return
      }

      const element = document.getElementById(id)
      if (element) {
        observer.observe(element)
      }
    })

    return () => {
      itemIds?.forEach((id) => {
        if (!id) {
          return
        }

        const element = document.getElementById(id)
        if (element) {
          observer.unobserve(element)
        }
      })
    }
  }, [itemIds])

  return activeId
}

interface TreeProps {
  tree: TableOfContents
  level?: number
  activeItem?: string | null
}

function Tree({ tree, level = 1, activeItem }: TreeProps) {
  return tree?.items?.length && level < 4 ? (
    <ul className={cn("m-0 list-none space-y-1", { "pl-4": level !== 1 })}>
      {tree.items.map((item, index) => {
        const isActive = item.url === `#${activeItem}`
        return (
          <li key={index}>
            <a
              href={item.url}
              className={cn(
                "block no-underline transition-colors hover:text-foreground",
                isActive
                  ? "font-semibold text-foreground"
                  : "text-sm text-muted-foreground",
                level === 1 && "text-base",
                level === 2 && "pl-2",
                level === 3 && "pl-4"
              )}
              onClick={(e) => {
                e.preventDefault()
                const targetId = item.url.replace("#", "")
                const element = document.getElementById(targetId)
                if (element) {
                  element.scrollIntoView({ behavior: "smooth", block: "start" })
                  // Update URL without scrolling
                  window.history.pushState(null, "", item.url)
                }
              }}
            >
              {item.title}
            </a>
            {item.items?.length ? (
              <Tree tree={item} level={level + 1} activeItem={activeItem} />
            ) : null}
          </li>
        )
      })}
    </ul>
  ) : null
}
