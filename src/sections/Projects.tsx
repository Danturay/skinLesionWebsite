
import darkSaasLandingPage from "@/assets/images/dark-saas-landing-page.png";
import lightSaasLandingPage from "@/assets/images/light-saas-landing-page.png";
import aiStartupLandingPage from "@/assets/images/ai-startup-landing-page.png";
import Image from "next/image";
import CheckIcon from "@/assets/icons/check-circle.svg"
import ArrowUpRightIcon from "@/assets/icons/arrow-up-right.svg"
import grainImage from '@/assets/images/grain.jpg';
import { SectionHeader } from "@/components/SectionHeader";
import { Card } from "@/components/Card";


const portfolioProjects = [
  {
    company: "University of Southampton",
    year: "2025",
    title: "Intergrating High Denisty fNIRS with AI for Lie Detection",
    results: [
      { title: "Achieved a First Class Honers for this 3rd Year Individual Project" },
      { title: "Conducted participant Trails to collect nurological primary data" },
      { title: "Implement CNN, LSTM and Hybrid Models" },
    ],
    linkName: "Read More",
    link: "https://youtu.be/4k7IdSLxh6w",
    image: darkSaasLandingPage,
  },
  {
    company: "Personal",
    year: "2025",
    title: "Fully Functional Skin Cancer Detection App Powered by AI (TODO)",
    results: [
      { title: "Used a React fronted with tailwind CSS and FlaskAPI" },
      { title: "Achieved accuracies of 97%" },
      { title: "User friendly interface with image upload and results display" },
    ],
    linkName: "View Live Site",
    link: "https://youtu.be/7hi5zwO75yc",
    image: lightSaasLandingPage,
  },
  {
    company: "Personal",
    year: "2025",
    title: "Portfolio Website",
    results: [
      { title: "Used Next.JS with Tailwind CSS too create an interactive portfolio" },
      { title: "Used OpenAI API to create a personliased chat able to awnser question about me" },
      { title: "Increased mobile traffic by 35%" },
    ],
    linkName: "View Live Site",
    link: "https://youtu.be/Z7I5uSRHMHg",
    image: aiStartupLandingPage,
  },
];

export const ProjectsSection = () => {
  return (
    <section className="pb-16 scroll-mt-20 lg:scroll-m-0 lg:py-24" id="projects">
      <div className="container">
        <SectionHeader 
          eyebrow="Real-world Results"
          title="Featured Projects"
          description="See how I transformed concepts into engaging digital experiences."
        />
        <div className="mt-10 md:mt-20 flex flex-col gap-20">
          {portfolioProjects.map((project, projectIndex)  => (

            <Card key={project.title} className="px-8 py-8 pt-8 pb-0 md:pt-12 md:px-10 lg:pt-16 lg:px-20 sticky" style={{
              top: `calc(85px + ${projectIndex * 40}px)`
            }}>
              <div className="lg:grid lg:grid-cols-2 lg:gap-16">
                <div className="lg:pb-16"> 
                  <div className="bg-gradient-to-r from-[#F3DFA7] to-[#C4A045] inline-flex gap-2 font-bold uppercase tracking-widest text-sm text-transparent bg-clip-text">
                    <span>{project.company}</span>
                    <span>&bull;</span>
                    <span>{project.year}</span>
                  </div>
                  <h3 className="font-serif text-2xl mt-2 md:mt-5 md:text-4xl">{project.title}</h3>
                  <hr className="border-t-2 border-white/5 mt-4" />
                  <ul className="flex flex-col gap-4 mt-4 md:mt-5">
                    {project.results.map((result) => (
                      <li className="flex gap-2 text-sm md:text-base text-white/50">
                        <CheckIcon className="size-5 md:size-6" />
                        <span>{result.title}</span>
                      </li>
                    ))}
                  </ul>
                  <a href={project.link}>
                    <button className="bg-white text-gray-950 h-12 w-full md:w-auto px-6 rounded-xl font-semibold inline-flex items-center justify-center gap-2 mt-8">
                      <span>{project.linkName}</span>
                      <ArrowUpRightIcon className="size-4" />
                    </button>
                  </a>
                </div>
                <div>
                  <Image src={project.image} alt={project.title} className="mt-8 -mb-4 md:-mb-0 lg:mt-0 lg:absolute lg:h-full lg:w-auto lg:max-w-none"/>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );

};
