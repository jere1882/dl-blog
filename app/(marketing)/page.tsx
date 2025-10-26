import Image from "next/image"
import Link from "next/link"
import { allPosts, allTags } from "contentlayer/generated"
import { compareDesc } from "date-fns"

import { siteConfig } from "@/config/site"
import { cn } from "@/lib/utils"
import { buttonVariants } from "@/components/ui/button"
import BlogPostList from "@/components/blog-post-list"
import { getTagsItems, TagGroup } from "@/components/tag-group"
import { TechCard } from "@/components/tech-card"

// Spotlight projects with full details
const SPOTLIGHT_PROJECTS = [
  {
    name: "Semantic Segmentation of Underwater Scenery",
    description:
      "Application of Vision Transformer and other architectures to semantic segmentation in underwater imagery.",
    link: "/blog/semantic-segmentation-of-underwater-scenery",
    tags: ["Computer Vision", "PyTorch"],
  },
  {
    name: "Document Hyper Resolution",
    description:
      "High-resolution document enhancement using deep learning to improve readability and preserve details.",
    link: "#", // TBD
    tags: ["Computer Vision", "PyTorch", "AWS EC2"],
  },
  {
    name: "View Synthesis: Indoor Scene Reconstruction",
    description:
      "Training Neural Radiance Field models on indoor images for realistic 3D scene reconstruction.",
    link: "/blog/indoor-nerf-reconstruction",
    tags: ["NeRF"],
  },
  {
    name: "Foundation Models for Astronomy",
    description:
      "Fine-tuning the AstroCLIP foundation model for galaxy k-correction estimation with image and spectral data.",
    link: "/blog/k-correction-via-foundation-model",
    tags: ["PyTorch", "Multimodal", "Foundation Models"],
  },
  {
    name: "Egyptian AI Lens",
    description:
      "AI-powered analysis of ancient Egyptian art using Google's Gemini to identify characters and translate hieroglyphs.",
    link: "/egyptian-ai-lens",
    tags: ["LLM", "Gemini", "AWS Lambda"],
  },
]

export default async function IndexPage() {
  const tagsItems = await getTagsItems(allTags)

  const posts = allPosts
    .filter((post) => post.publish)
    .sort((a, b) => {
      return compareDesc(new Date(a.date), new Date(b.date))
    })
    .slice(0, 4)

  return (
    <>
      <section className="container flex flex-col-reverse items-center justify-center gap-6 pb-8 pt-6 sm:flex-row md:pb-12 md:pt-10 lg:pb-24 lg:pt-16">
        <div className="flex flex-col items-center gap-4 sm:items-start">
          <h1 className="text-3xl font-bold leading-[1.1] tracking-tighter sm:text-5xl md:text-6xl">
            Jeremias Rodriguez
          </h1>
          <h2 className="text-lg font-semibold tracking-tighter sm:text-2xl md:text-3xl">
            Machine Learning in Action
          </h2>
          <p className=" text-center leading-normal text-muted-foreground sm:text-start sm:text-xl sm:leading-8">
            Welcome to my personal blog, where I explore the depths of AI, one
            project at a time.
          </p>
          <div className="flex gap-4">
            <Link href="/blog" className={cn(buttonVariants({ size: "lg" }))}>
              Visit the blog
            </Link>
            <Link
              href="https://github.com/jere1882/"
              className={cn(buttonVariants({ size: "lg", variant: "outline" }))}
            >
              Github
            </Link>
            <Link
              href="https://www.linkedin.com/in/jere-rodriguez/"
              target="_blank"
              rel="noopener noreferrer"
              className={cn(buttonVariants({ size: "lg", variant: "outline" }))}
            >
              LinkedIn
            </Link>
          </div>
        </div>
        <div className="size-36 relative mx-16 flex shrink-0 overflow-hidden rounded-full bg-gradient-to-b from-primary to-blue-200 shadow-lg ring-4 ring-primary/80">
          <Image
            src="/profile_picture.png"
            alt={`${siteConfig.name}'s Picture`}
            className="w-ful aspect-square h-full"
            width={250}
            height={250}
          />
        </div>
      </section>
      <hr className="container" />
      <section
        id="features"
        className="container space-y-6 py-8 md:py-12 lg:py-24"
      >
        <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-4 text-center">
          <h2 className="text-3xl font-semibold sm:text-3xl md:text-6xl">
            Spotlight
          </h2>
          <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
            {"Selected projects from my main areas of interest."}
          </p>
        </div>
        <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:max-w-[90rem] lg:grid-cols-3 xl:grid-cols-5">
          {SPOTLIGHT_PROJECTS.map((project) => (
            <TechCard
              key={project.name}
              name={project.name}
              description={project.description}
              link={project.link}
              tags={project.tags}
            />
          ))}
        </div>
      </section>
      <hr className="container" />
      <section className="container space-y-8 py-8 md:max-w-4xl md:py-12 lg:py-16">
        <h2 className="mb-4 scroll-m-20 pb-1 text-center text-2xl font-semibold tracking-tight first:mt-0 md:text-5xl">
          {"Recent Posts"}
        </h2>
        <BlogPostList posts={posts} />
        {allPosts.length > posts.length && (
          <div className="flex justify-end text-base font-medium leading-6">
            <Link
              href="/blog"
              className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
              aria-label="all posts"
            >
              All Posts &rarr;
            </Link>
          </div>
        )}
      </section>
      <hr className="container" />
      <section className="container space-y-8 py-8 md:max-w-4xl md:py-12 lg:py-16">
        <h2 className="mb-4 scroll-m-20 pb-1 text-center text-2xl font-semibold tracking-tight first:mt-0 md:text-5xl">
          All topics
        </h2>
        <TagGroup tagsItems={tagsItems} />
      </section>
    </>
  )
}
