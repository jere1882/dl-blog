import { Metadata } from "next"

export const metadata: Metadata = {
  title: "CV | Jeremias Rodriguez",
  description:
    "Curriculum Vitae of Jeremias Rodriguez - Machine Learning Engineer and Software Engineer with expertise in computer vision and robotics.",
}

export default function CVPage() {
  return (
    <div className="container mx-auto max-w-5xl px-4 py-8">
      <div className="space-y-8">
        {/* Header */}
        <div className="space-y-4 border-b pb-8 text-center">
          <h1 className="text-4xl font-bold">Jeremias Rodriguez</h1>
          <p className="text-xl text-muted-foreground">
            Machine Learning Engineer • Software Engineer
          </p>
          <div className="flex justify-center space-x-4 text-sm">
            <a href="mailto:jeremiaslcc@gmail.com" className="hover:underline">
              jeremiaslcc@gmail.com
            </a>
            <span>•</span>
            <a
              href="https://www.jeremias-rodriguez.com/"
              className="hover:underline"
            >
              jeremias-rodriguez.com
            </a>
            <span>•</span>
            <a
              href="https://www.linkedin.com/in/jere-rodriguez/"
              className="hover:underline"
            >
              LinkedIn
            </a>
          </div>
        </div>

        <div className="grid gap-8 md:grid-cols-3">
          {/* Left Column - Main Content */}
          <div className="space-y-8 md:col-span-2">
            {/* Experience */}
            <section>
              <h2 className="mb-6 border-b pb-2 text-2xl font-bold">
                Experience
              </h2>

              <div className="space-y-6">
                {/* Homevision */}
                <div>
                  <div className="mb-2 flex items-start justify-between">
                    <h3 className="text-xl font-semibold">Homevision</h3>
                    <span className="text-sm text-muted-foreground">
                      January 2025 - present
                    </span>
                  </div>
                  <h4 className="mb-1 text-lg font-medium text-blue-600">
                    Sr. Machine Learning Engineer - Tech Lead
                  </h4>
                  <p className="mb-3 text-sm text-muted-foreground">
                    Full-time remote contractor, Argentina
                  </p>
                  <ul className="list-inside list-disc space-y-1 text-sm">
                    <li>
                      Evaluation and integration of different computer vision
                      and large language models into a complex, multi-modal
                      document understanding system
                    </li>
                    <li>
                      Tasks include low quality document understanding,
                      subjective and biased language detection, automated QC,
                      etc. Responsibilities range from data science to
                      deployment and production monitoring
                    </li>
                  </ul>
                </div>

                {/* iRobot */}
                <div>
                  <div className="mb-2 flex items-start justify-between">
                    <h3 className="text-xl font-semibold">
                      iRobot Corporation
                    </h3>
                    <span className="text-sm text-muted-foreground">
                      April 2018 - January 2025
                    </span>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <h4 className="text-lg font-medium text-blue-600">
                        Sr. Software Engineer
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        Sept 2021 - January 2025 • Full-time employee - London,
                        UK
                      </p>
                    </div>
                    <div>
                      <h4 className="text-lg font-medium text-blue-600">
                        Software Engineer
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        April 2019 - September 2021 • Full-time remote
                        contractor - Rosario, Argentina
                      </p>
                    </div>
                    <div>
                      <h4 className="text-lg font-medium text-blue-600">
                        Robotics Intern
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        April 2018 - April 2019 • Research Intern - Pasadena,
                        CA, US
                      </p>
                    </div>
                  </div>

                  <ul className="mt-3 list-inside list-disc space-y-1 text-sm">
                    <li>
                      Trained and maintained a semantic segmentation deep
                      learning model deployed in millions of Roomba c10 and j9
                      robots. Single-handedly prototyped a 3D scene
                      reconstruction system for Roomba based on NERFs (PyTorch,
                      Python)
                    </li>
                    <li>
                      Independently designed, implemented, and monitored a novel
                      long-term mapping system for Roomba c10 and j9, which
                      accurately tracks floor type and carpets using computer
                      vision
                    </li>
                    <li>
                      Core member of the engineering team that developed a novel
                      camera-less GraphSLAM method for the Roomba i3, i4, and
                      i5, which sold <strong>millions</strong> of units
                      worldwide
                    </li>
                    <li>
                      Created and maintained dashboards to monitor system
                      metrics and performance for millions of Roomba i3 units in
                      production, alpha and beta testing phases (Mode, Python)
                    </li>
                  </ul>
                </div>

                {/* Plantium */}
                <div>
                  <div className="mb-2 flex items-start justify-between">
                    <h3 className="text-xl font-semibold">Plantium SA</h3>
                    <span className="text-sm text-muted-foreground">
                      November 2017 - April 2018
                    </span>
                  </div>
                  <h4 className="mb-1 text-lg font-medium text-blue-600">
                    Software Engineer
                  </h4>
                  <p className="mb-3 text-sm text-muted-foreground">
                    Rosario, Argentina
                  </p>
                  <ul className="list-inside list-disc space-y-1 text-sm">
                    <li>
                      Joined the engineering team that released the SBOX7 and
                      SBOX11 products, used for autonomous precision agriculture
                      (QT / C++)
                    </li>
                  </ul>
                </div>
              </div>
            </section>

            {/* Personal */}
            <section>
              <h2 className="mb-6 border-b pb-2 text-2xl font-bold">
                About Me, Outside Work
              </h2>
              <div className="space-y-3 text-sm">
                <div className="flex items-start space-x-3">
                  <span className="text-lg">🌍</span>
                  <p>
                    My passion is traveling the world, especially to
                    off-the-beaten-path destinations. My top 3 favorite trips
                    have been Egypt, Japan, and Scotland.
                  </p>
                </div>

                <div className="flex items-start space-x-3">
                  <span className="text-lg">☕</span>
                  <p>
                    I deeply enjoy working from coffee shops, somehow ambient
                    noise helps me focus in ways that no fancy office desk can
                    match.
                  </p>
                </div>

                <div className="flex items-start space-x-3">
                  <span className="text-lg">🤿</span>
                  <p>
                    <strong>Scuba diving:</strong> I am a certified advanced
                    diver, and when possible, diving adds a fascinating
                    dimension to all my travels.
                  </p>
                </div>

                <div className="flex items-start space-x-3">
                  <span className="text-lg">📚</span>
                  <p>
                    I&apos;m an avid reader of fantasy and sci-fi books. My top
                    3 series are <em>The Wheel of Time</em>,{" "}
                    <em>Harry Potter</em>, and <em>The Stormlight Archive</em>.
                  </p>
                </div>

                <div className="flex items-start space-x-3">
                  <span className="text-lg">🏐</span>
                  <p>
                    I&apos;ve been playing volleyball since I was 13. I&apos;m a
                    proud member of the Italian team in my city, where I play as
                    wing spiker.
                  </p>
                </div>

                <div className="flex items-start space-x-3">
                  <span className="text-lg">🇦🇷🇮🇹</span>
                  <p>Proudly holding dual Argentinian/Italian citizenship.</p>
                </div>
              </div>
            </section>
          </div>

          {/* Right Column - Skills & Education */}
          <div className="space-y-8">
            {/* Skills */}
            <section>
              <h2 className="mb-4 border-b pb-2 text-xl font-bold">Skills</h2>
              <div className="space-y-4 text-sm">
                <div>
                  <h3 className="mb-2 font-semibold">Machine Learning</h3>
                  <p className="mb-2">
                    Particularly experienced in <strong>computer vision</strong>{" "}
                    (object detection, semantic segmentation, scene
                    reconstruction) for embedded devices, and{" "}
                    <strong>modern LLM technologies</strong> (especially
                    Gemini). Python, PyTorch, Docker.
                  </p>
                  <div className="space-y-1">
                    <p className="text-xs text-blue-600">
                      <a
                        href="https://www.jeremias-rodriguez.com/blog/semantic-segmentation-of-underwater-scenery"
                        className="hover:underline"
                      >
                        Project: Semantic segmentation of underwater scenes
                      </a>
                    </p>
                    <p className="text-xs text-blue-600">
                      <a
                        href="https://www.jeremias-rodriguez.com/egyptian-ai-lens"
                        className="hover:underline"
                      >
                        Project: Egyptian AI Lens - LLM Powered ancient art
                        analysis
                      </a>
                    </p>
                  </div>
                </div>

                <div>
                  <h3 className="mb-2 font-semibold">Software Engineering</h3>
                  <p>
                    Extensive experience in object oriented programming,
                    particularly in robotics. 6+ years coding in C++ on Linux.
                    Very familiar with design patterns, unit testing and good
                    code practices. Go, Python.
                  </p>
                </div>

                <div>
                  <h3 className="mb-2 font-semibold">Data Science</h3>
                  <p>
                    Experienced in creating dashboards to track performance and
                    detect issues. (Python, SQL, Mode, Datadog)
                  </p>
                </div>

                <div>
                  <h3 className="mb-2 font-semibold">Teamwork Tools</h3>
                  <p>
                    Very familiar with Jira, Git, Bitbucket, Weights&Biases,
                    Confluence. Experienced with agile methodologies.
                  </p>
                </div>

                <div>
                  <h3 className="mb-2 font-semibold">Languages</h3>
                  <p>
                    Proficient in English (Cambridge Proficiency Grade A; IELTS
                    Band 8). Good communicator.
                  </p>
                </div>
              </div>
            </section>

            {/* Education */}
            <section>
              <h2 className="mb-4 border-b pb-2 text-xl font-bold">
                Education
              </h2>
              <div>
                <h3 className="font-semibold">
                  Master&apos;s Degree in Computer Science
                </h3>
                <p className="text-sm text-muted-foreground">2014-2022</p>
                <p className="text-sm">
                  Universidad Nacional de Rosario, Argentina
                </p>
                <p className="mt-1 text-xs">
                  Master&apos;s grade average: 9.50/10 (2017-2022)
                </p>
                <p className="text-xs">
                  Bachelor&apos;s grade average: 9.52/10 (2014-2017)
                </p>
              </div>
            </section>

            {/* Awards */}
            <section>
              <h2 className="mb-4 border-b pb-2 text-xl font-bold">Awards</h2>
              <ul className="list-inside list-disc text-sm">
                <li>Second highest GPA amongst all 2022 FCEIA graduates</li>
              </ul>
            </section>

            {/* Publications */}
            <section>
              <h2 className="mb-4 border-b pb-2 text-xl font-bold">
                Publications
              </h2>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="font-medium">
                    Trajectory-Based SLAM for Indoor Mobile Robots with Limited
                    Sensing Capabilities
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Chen, Rodriguez, Karimian, Okerholm Hutlin et al (iRobot) -
                    IROS 2023 (WeAT14.7)
                  </p>
                </div>
                <div>
                  <p className="font-medium">
                    Deep K-Correct: Estimating K-Corrections and Absolute
                    Magnitudes from Galaxy Images
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Fine tuning AstroCLIP foundation model. Paris workshop on
                    Bayesian Deep Learning for Cosmology and Time Domain
                    Astrophysics. Rodriguez, Dominguez. 2025
                  </p>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}
