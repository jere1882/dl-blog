import { UserConfig } from "../types/index"

const config: UserConfig = {
  title: "Jeremias Rodriguez",
  description:
    "Hi, I'm Jeremias! Passionate about deep learning, I share insights from my journey, exploring topics like computer vision, transformers, and interpretability.",
  name: "Jeremias Rodriguez",
  url: "https://www.jeremiasrodriguez.com/",
  favicon: "",
  profilePicture: "/profile_picture.png",

  links: {
    linkedin: "https://www.linkedin.com/in/jeremias-rodriguez/",
    github: "https://github.com/jere1882",
    fran_github: "https://github.com/FranciscoMoretti/Site",
  },

  editLinkRoot: "https://github.com/jere1882/site/edit/main/site/content",
  showEditLink: false,
  showToc: true,
  showSidebar: false,

  contentExclude: ["templates"],

  comments: {
    provider: "giscus", // supported providers: giscus, utterances, disqus
    pages: ["blog"], // page directories where we want commments
    config: {
      repo: process.env.NEXT_PUBLIC_GISCUS_REPO,
      repositoryId: process.env.NEXT_PUBLIC_GISCUS_REPOSITORY_ID,
      category: process.env.NEXT_PUBLIC_GISCUS_CATEGORY,
      categoryId: process.env.NEXT_PUBLIC_GISCUS_CATEGORY_ID,
    },
  },
  analytics: "G-RQWLTRWBS2",
  navLinks: [
    {
      title: "📄 Resume",
      href: "/cv",
    },
    {
      title: "🏺 Egyptian AI Lens",
      href: "/egyptian-ai-lens",
    },
    //{ href: "/code-tips", title: "Code Tips" },
    //{ href: "/blog", title: "Blog" },
  ],
  social: [
    { label: "github", href: "https://github.com/flowershow/flowershow" },
    { label: "discord", href: "https://discord.gg/cPxejPzpwt" },
  ],
  search: {
    provider: "algolia",
    config: {
      appId: process.env.NEXT_PUBLIC_DOCSEARCH_APP_ID,
      apiKey: process.env.NEXT_PUBLIC_DOCSEARCH_API_KEY,
      indexName: process.env.NEXT_PUBLIC_DOCSEARCH_INDEX_NAME,
    },
  },
}

export default config
