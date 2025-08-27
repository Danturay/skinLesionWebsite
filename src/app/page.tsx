"use client";
import { useScroll, useTransform, motion } from "framer-motion";
import { useRef } from "react";

import { NeuronsBackground } from "@/components/NeuronsBackground";
import { AboutSection } from "@/sections/About";
import { ContactSection } from "@/sections/Contact";
import { Footer } from "@/sections/Footer";
import { Header } from "@/sections/Header";
import { HeroSection } from "@/sections/Hero";
import Navbar from "@/sections/Navbar";
import { UploadSection } from "@/sections/Upload";

export default function Home() {

  return (
    <div>
      <Navbar />
      <HeroSection />
      <UploadSection />
    </div>
  );
}
