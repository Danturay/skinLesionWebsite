import Image from "next/image";
import doctorImage from "@/assets/icons/doctor.png";
import ArrowDown from "@/assets/icons/arrow-down.svg";


export const HeroSection = () => {
  return (
    <section className="container min-h-screen flex flex-col py-20 px-2" id="home">
  {/* This grid keeps the 2-column layout as is */}
  <div className="grid grid-cols-2 gap-20 items-center flex-grow">
    <div className="flex flex-col">
      <h1 className="font-serif text-6xl tracking-wide">Your Skin,</h1>
      <h1 className="font-serif text-3xl md:text-6xl tracking-wide py-2 mb-4">Our Priority</h1>
      <p className="text-xl">
        We Provide a skin lesion detection service that can detect a range of skin lesions powered by AI. Worried about a skin Lesion? Just upload a photo and get results within seconds. Free, no Sign up needed
      </p>
    </div>
    <div className="relative w-[200px] h-64 ml-20">
      <Image src={doctorImage} alt="Person behind laptop" fill className="object-contain rounded-full" />
    </div>
  </div>

  {/* Buttons container below grid */}
  <div className="flex flex-row items-center space-x-10 justify-center">
    <button className="bg-[#5A9BD5] text-white inline-flex items-center gap-2 px-6 h-12 rounded-xl border border-transparent shadow-md transition-all duration-300 ease-in-out hover:bg-[#4a81c3] hover:shadow-lg hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-[#5A9BD5] focus:ring-offset-2">
  <span className="font-semibold">Upload Photo</span>
  <ArrowDown className="w-4 h-4" />
</button>

<button className="bg-[#5A9BD5] text-white inline-flex items-center gap-2 px-6 h-12 rounded-xl border border-transparent shadow-md transition-all duration-300 ease-in-out hover:bg-[#4a81c3] hover:shadow-lg hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-[#5A9BD5] focus:ring-offset-2">
  <span className="font-semibold">Learn More</span>
  <ArrowDown className="w-4 h-4" />
</button>
  </div>
</section>


      /* <div className="container">
        <div className="flex flex-col items-center">
          <Image src={meTemp} alt="Person behind laptop" className="w-36 md:w-48"/>
       </div>
       <div className="flex justify-center mt-4">
  <div className="font-serif text-3xl md:text-4xl text-center tracking-wide flex flex-col md:flex-row items-center gap-2 md:gap-4">
  {/* Line 1: Student • AI Developer */
/* <div className="flex gap-3 md:gap-4">
    <span>Student</span>
    <span className="text-[#C4A045]">&bull;</span>
    <span>AI Developer</span>
  </div>
  <div className="flex gap-3 md:hidden">
    <span className="text-[#C4A045]">&bull;</span>
    <span>Researcher</span>
  </div>
  <div className="hidden md:flex gap-4">
    <span className="text-[#C4A045]">&bull;</span>
    <span>Researcher</span>
  </div>
</div>

</div>
       <p className="mt-4 text-center text-white/60 md:text-lg">As a dedicated student in artificial intelligence.</p>
       <div className="flex flex-col md:flex-row justify-center items-center mt-8 gap-4">
        <a href="#projects">
          <button className="inline-flex items-center gap-2 border border-white/15 px-6 h-12 rounded-xl transition-all duration-300 hover:bg-white/10 hover:border-white/30 hover:scale-105">
            <span className="font-semibold">Explore My Work</span>
            <ArrowDown className="size-4" />
          </button>
        </a>
        <button className="inline-flex items-center gap-2 border-white bg-white text-gray-900 h-12 px-6 rounded-xl transition-all duration-300 hover:bg-gray-100 hover:shadow-md hover:scale-105">
          <span>👋</span>
          <span className="font-semibold">Lets Connect</span>
        </button>
       </div>
      </div> */
    /* </section> */
  )
};
