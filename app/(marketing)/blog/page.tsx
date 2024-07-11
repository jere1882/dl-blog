import { allPosts } from "contentlayer/generated"
import { compareDesc } from "date-fns"

import { PostCard } from "@/components/post-card"

export const revalidate = 60

export const metadata = {
  title: "Blog",
  description:
    "Welcome to Machine Learning in Action! In my personal blog, you can dive into the fascinating world of deep learning, one project at a time!",
}

export default async function BlogPage() {
  const posts = allPosts
    .filter((post) => post.publish)
    .sort((a, b) => {
      return compareDesc(new Date(a.date), new Date(b.date))
    })

  return (
    <div className="container max-w-4xl py-6 lg:py-10">
      <div className="flex flex-col items-start gap-4 md:gap-8">
        <div className="flex flex-1 flex-col space-y-4">
          <h1 className="inline-block text-4xl font-extrabold tracking-tight lg:text-5xl">
            Machine Learning in Action Blog!
          </h1>
          <h2 className="inline-block text-2xl font-bold tracking-tight lg:text-3xl">
            Exploring the depths of AI, one project at a time.
          </h2>
          <p className="text-xl text-muted-foreground">
            {
              "Welcome to Machine Learning in Action! Explore the fascinating world of deep learning through my personal blog, where I showcase different projects I've worked on, share insights gained, and provide modern machine learning tips."
            }
          </p>
        </div>
      </div>
      <hr className="my-8" />
      {posts?.length ? (
        <div className="grid gap-10 sm:grid-cols-2">
          {posts.map((post, index) => PostCard({ post, index }))}
        </div>
      ) : (
        <p>No posts published.</p>
      )}
    </div>
  )
}
