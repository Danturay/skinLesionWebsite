'use client';
import { SectionHeader } from "@/components/SectionHeader";
import { Card } from "@/components/Card";
import { TechIcon } from "@/components/TechIcon";
import logo from "@/assets/images/logo.png";
import Image from "next/image";
import JavascriptIcon from "@/assets/icons/square-js.svg";
import HtmlIcon from "@/assets/icons/html5.svg";
import CssIcon from "@/assets/icons/css3.svg";
import ReactIcon from "@/assets/icons/react.svg";
import ChromeIcon from "@/assets/icons/chrome.svg";
import GithubIcon from "@/assets/icons/github.svg";
import JavaIcon from "@/assets/icons/java.svg";
import SqlIcon from "@/assets/icons/sql.svg";
import TensorFlowIcon from "@/assets/icons/tensorflow.svg";
import HaskellIcon from "@/assets/icons/haskell.svg";
import TailwindIcon from "@/assets/icons/tailwind.svg";
import PythonIcon from "@/assets/icons/python.svg";
import { CardHeader } from "@/components/CardHeader";
import { ToolboxItems } from "@/components/ToolboxItems";
import { motion } from "framer-motion";
import { useRef } from "react";
import { i } from "framer-motion/client";
import gradCap from "@/assets/icons/grad.svg";
import book from "@/assets/icons/book.svg";

const toolboxItems = [
  {
    title: 'JavaScript',
    iconType: JavascriptIcon,
  },
   {
    title: 'React',
    iconType: ReactIcon,
  },
   {
    title: 'Github',
    iconType: GithubIcon,
  },
  {
    title: 'Python',
    iconType: PythonIcon
  },
   {
    title: 'Java',
    iconType: JavaIcon,
  },
   {
    title: 'SQL',
    iconType: SqlIcon,
  },
   {
    title: 'TensorFlow',
    iconType: TensorFlowIcon,
  },
   {
    title: 'Haskell',
    iconType: HaskellIcon,
  },
   {
    title: 'Tailwind CSS',
    iconType: TailwindIcon,
  },
]

const hobbies = [
  {
    title: "Muay Thai",
    emoji: "🥊",
    left: "5%",
    top: "5%"
  },
  {
    title: "Rugby",
    emoji: "🏉",
    left: "50%",
    top: "5%"
  },
  {
    title: "Rubik's Cube",
    emoji: "🧠",
    left: "10%",
    top: "35%"
  },
  {
    title: "Piano",
    emoji: "🎹",
    left: "35%",
    top: "45%"
  },
  {
    title: "Basketball",
    emoji: "🏀",
    left: "65%",
    top: "25%"
  },
  {
    title: "Reaserch",
    emoji: "💻",
    left: "5%",
    top: "65%"
  },
  {
    title: "Gaming",
    emoji: "🎮",
    left: "45%",
    top: "70% "
  },
]

export const AboutSection = () => {
  const constraintRef = useRef(null);
  return (
    <section className="py-20 lg:py-28" id="about">
      <div className="container">
        <SectionHeader 
          eyebrow="About Me"
          title="Crafting Digital Experiences"
          description="I specialize in creating engaging, user-friendly websites that drive results and enhance brand presence."
        />
        <div className="mt-20 flex flex-col gap-8">
          <div className="grid grid-cols-1 gap-8 md:grid md:grid-cols-5 md:gap-8 lg:grid-cols-3">
            <Card className="h-[320px] md:col-span-2 lg:col-span-1">
              <CardHeader title="Education"/>
              <div>
                <div className="flex flex-row gap-3 mb-2">
                  <TechIcon component={gradCap} className="size-10" />
                  <div className="mx-auto">
                    <h4 className="uppercase font-semibold tracking-widest bg-gradient-to-r from-[#F3DFA7] to-[#C4A045] text-center text-transparent bg-clip-text">University of Southampton</h4>
                    <p className="text-sm text-center">MEng Computer Science with Artifical Intelligence | 2:1</p>
                  </div>
                </div>
                <hr className="border-t-2 border-white/5 mt-4" />
                <div className="flex items-start gap-3 mt-4">
                  <TechIcon component={book} className="size-7" />
                  <div>
                    <h4 className="uppercase font-semibold tracking-widest bg-gradient-to-r from-[#F3DFA7] to-[#C4A045] text-center text-transparent bg-clip-text">Beths Grammar School</h4>
                    <ul className="text-sm text-center">
                      <li>Computer Science | A*</li>
                      <li>Maths | A*</li>
                      <li>Physics | A</li>
                    </ul>
                  </div>
                </div>
              </div>

            </Card>
            <Card className="h-[320px] p-0 md:col-span-3 lg:col-span-2">
              <CardHeader title="My Toolbox" description="l" className="px-6 pt-6" />
              <ToolboxItems toolboxItems={toolboxItems} className="mt-6" itemsWrapperClassName="animate-move-left [animation-duration:35s]" />
              <ToolboxItems toolboxItems={toolboxItems} className="mt-6" itemsWrapperClassName="animate-move-right [animation-duration:20s]"/>
            </Card>
          </div>
          <div className="grid grid-cols-1 gap-8 md:grid md:grid-cols-5 md:gap-8 lg:grid-cols-3">
            <Card className="h-[320px] p-0 flex flex-col md:col-span-3 lg:col-span-2">
              <CardHeader title="Beyond the Code" className="px-6 py-6" />
              <div className="relative flex-1" ref={constraintRef}>
                {hobbies.map(hobby => (
                  <motion.div key={hobby.title} className="inline-flex items-center gap-2 px-6 bg-gradient-to-r from-[#F3DFA7] to-[#C4A045] rounded-full py-1.5 absolute"
                      style={{ left: hobby.left, top: hobby.top }} drag dragConstraints={constraintRef}>
                    <span className="font-medium text-gray-950">{hobby.title}</span>
                    <span>{hobby.emoji}</span>
                  </motion.div>
                ))}
              </div>
            </Card>
            <Card className="h-[320px] md:col-span-2 lg:col-span-1"> 
              <CardHeader title="Certifications" />
     
                <h4 className="uppercase font-semibold tracking-widest bg-gradient-to-r from-[#F3DFA7] to-[#C4A045] text-transparent bg-clip-text py-2">DeepLearning.AI</h4>
              <ul className="list-disc ml-3">
                <li>AI for Medical Specialisation (In Progress)</li>
              </ul>
                <h4 className="uppercase font-semibold tracking-widest bg-gradient-to-r from-[#F3DFA7] to-[#C4A045] text-transparent bg-clip-text py-2">Amazon</h4>
              <ul className="list-disc ml-3">  
                <li>AWS Cloud Practitioner (To Do)</li>
                <li>AWS Machine Learning Engineer - Associate (TO DO)</li>
              </ul>
            </Card>
          </div>
        </div>
      </div>
    </section>
  ) ;
};
