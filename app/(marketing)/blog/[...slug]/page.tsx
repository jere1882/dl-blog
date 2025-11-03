import { existsSync, readdirSync, readFileSync } from "fs"
import path from "path"
import { Metadata } from "next"
import Image from "next/image"
import Link from "next/link"
import { notFound } from "next/navigation"
import { allAuthors, allPosts } from "contentlayer/generated"

import { routepathToSlug } from "@/lib/path"
import { getTableOfContents, TableOfContents } from "@/lib/toc"
import { absoluteUrl, cn, formatDate } from "@/lib/utils"
import { BlogTableOfContents } from "@/components/blog-toc"
import { Icons } from "@/components/icons"
import { Mdx } from "@/components/mdx"
import { upsertPost } from "@/app/(marketing)/actions"

import "@/styles/mdx.css"

import { buttonVariants } from "@/components/ui/button"

export const revalidate = 60

interface PostPageProps {
  params: {
    slug: string[]
  }
}

async function getPostFromParams(params) {
  const slug = params?.slug?.join("/")
  const post = allPosts.find((post) => routepathToSlug(post.routepath) === slug)

  if (!post) {
    null
  }

  return post
}

export async function generateMetadata({
  params,
}: PostPageProps): Promise<Metadata> {
  const post = await getPostFromParams(params)

  if (!post) {
    return {}
  }

  const url = process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3002"

  const ogUrl = new URL(`${url}/api/og`)
  ogUrl.searchParams.set("heading", post.title)
  ogUrl.searchParams.set("tags", post.tags.join("|"))
  ogUrl.searchParams.set("type", "Blog Post")
  ogUrl.searchParams.set("mode", "light")

  return {
    title: {
      absolute: post.title,
    },
    description: post.description,
    authors: post.authors.map((author) => ({
      name: author,
    })),
    openGraph: {
      title: post.title,
      description: post.description,
      type: "article",
      url: absoluteUrl(post.routepath),
      images: [
        {
          url: ogUrl.toString(),
          width: 1200,
          height: 630,
          alt: post.title,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: post.title,
      description: post.description,
      images: [ogUrl.toString()],
    },
  }
}

export async function generateStaticParams(): Promise<
  PostPageProps["params"][]
> {
  const promises = allPosts.map(async (post) => {
    const { slug } = post
    try {
      return await upsertPost(slug)
    } catch (error) {
      console.log(error)
    }
  })

  if (process.env.NODE_ENV !== "production") {
    await Promise.all(promises).catch((error) => {
      console.log("Error:", error)
    })
  }

  return allPosts.map((post) => ({
    slug: [post.slug],
  }))
}

export default async function PostPage({ params }: PostPageProps) {
  const post = await getPostFromParams(params)

  if (!post) {
    notFound()
  }

  const authors = post.authors.map((author) =>
    allAuthors.find(({ routepath }) => routepath === `/authors/${author}`)
  )

  // Read the raw markdown file to generate TOC
  let toc: TableOfContents | null = null
  try {
    const contentDir = path.join(process.cwd(), "content")
    const blogDir = path.join(contentDir, "blog")
    let filePath: string | undefined

    // Search for file by matching slug in frontmatter (most reliable method)
    if (existsSync(blogDir)) {
      const files = readdirSync(blogDir).filter(
        (f: string) => f.endsWith(".md") || f.endsWith(".mdx")
      )

      // Try to find file by reading frontmatter and matching slug
      for (const file of files) {
        try {
          const fullPath = path.join(blogDir, file)
          const content = readFileSync(fullPath, "utf-8")
          const frontmatterMatch = content.match(/^slug:\s*(.+)$/m)
          if (frontmatterMatch && frontmatterMatch[1].trim() === post.slug) {
            filePath = fullPath
            break
          }
        } catch {
          continue
        }
      }
    }

    if (!filePath) {
      console.error(
        `TOC: Could not find markdown file for post with slug: ${post.slug}`
      )
    } else {
      const rawContent = readFileSync(filePath, "utf-8")
      // Remove frontmatter for TOC generation
      const contentWithoutFrontmatter = rawContent.replace(
        /^---\n[\s\S]*?\n---\n/,
        ""
      )
      toc = await getTableOfContents(contentWithoutFrontmatter)
      // Debug: log TOC generation result
      if (!toc || !toc.items || toc.items.length === 0) {
        console.log(
          `TOC: No items found for post ${post.slug} (file: ${path.basename(
            filePath
          )})`
        )
      } else {
        console.log(
          `TOC: Found ${toc.items.length} top-level items for post ${post.slug}`
        )
      }
    }
  } catch (error) {
    console.error(`TOC: Error generating TOC for post ${post.slug}:`, error)
    // TOC is optional, continue without it
  }

  return (
    <article className="container relative max-w-3xl py-6 lg:py-10">
      <Link
        href="/blog"
        className={cn(
          buttonVariants({ variant: "ghost" }),
          "absolute left-[-200px] top-14 hidden xl:inline-flex"
        )}
      >
        <Icons.chevronLeft className="mr-2 h-4 w-4" />
        See all posts
      </Link>
      <div>
        {post.date && (
          <div className="flex space-x-4 text-sm text-muted-foreground">
            <time dateTime={post.date} className="block">
              Published on {formatDate(post.date)}
            </time>
          </div>
        )}
        <h1 className="mt-2 inline-block text-4xl font-extrabold leading-tight lg:text-5xl">
          {post.title}
        </h1>
        {authors?.length ? (
          <div className="mb-2 mt-4 flex space-x-4">
            {authors.map((author) =>
              author ? (
                <Link
                  key={author._id}
                  href={`https://twitter.com/${author.twitter}`}
                  className="flex items-center space-x-2 text-sm"
                >
                  <Image
                    src={author.avatar}
                    alt={author.title}
                    width={42}
                    height={42}
                    className="rounded-full"
                  />
                  <div className="flex-1 text-left leading-tight">
                    <p className="font-medium">{author.title}</p>
                    <p className="text-[12px] text-muted-foreground">
                      @{author.twitter}
                    </p>
                  </div>
                </Link>
              ) : null
            )}
          </div>
        ) : null}
      </div>
      {post.image && (
        <Image
          src={post.image}
          alt={post.title}
          width={720}
          height={405}
          className="my-8 rounded-md border bg-muted transition-colors"
          priority
        />
      )}
      <hr className="my-4" />
      {toc && (
        <div className="my-8">
          <BlogTableOfContents toc={toc} />
        </div>
      )}
      <Mdx code={post.body.code} />
      <hr className="my-4" />
      <div className="flex justify-center py-6 lg:py-10">
        <Link href="/blog" className={cn(buttonVariants({ variant: "ghost" }))}>
          <Icons.chevronLeft className="mr-2 h-4 w-4" />
          See all posts
        </Link>
      </div>
    </article>
  )
}
