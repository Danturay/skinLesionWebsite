import Image from "next/image";
import doctorImage from "@/assets/images/doctor.png";
import ArrowDown from "@/assets/icons/arrow-down.svg";
import ArrowUpRight from "@/assets/icons/arrow-up-right.svg";

export const HeroSection = () => {
  return (
    <section
      className="container min-h-screen flex flex-col justify-center py-20 px-4 md:px-8"
      id="home"
    >
      {/* Responsive grid: 1 column on mobile, 2 on md+ */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-10 md:gap-20 items-center flex-grow">
        {/* Left side: Text */}
        <div className="flex flex-col justify-center text-center md:text-left">
          <h1 className="font-serif text-5xl md:text-6xl tracking-wide">
            Your Skin,
          </h1>
          <h1 className="font-serif text-3xl md:text-6xl tracking-wide py-2 mb-4">
            Our Priority
          </h1>
          <p className="text-lg md:text-xl text-gray-700 leading-relaxed">
            We provide a skin lesion detection service powered by AI that can
            detect a range of skin conditions. Worried about a lesion? Just
            upload a photo and get results within seconds. Free, no sign-up
            required.
          </p>
        </div>

        {/* Right side: Image */}
        <div className="relative w-full h-64 sm:h-80 md:h-96 flex justify-center md:justify-end items-center">
          <div className="relative w-48 h-48 sm:w-56 sm:h-56 md:w-64 md:h-64 rounded-full overflow-hidden">
            <Image
              src={doctorImage}
              alt="Person behind laptop"
              fill
              className="object-contain"
            />
          </div>
        </div>
      </div>

      {/* Buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-center mt-10 space-y-4 sm:space-y-0 sm:space-x-6">
        <a
          href="#upload"
          className="bg-[#5A9BD5] text-white inline-flex items-center gap-2 px-6 h-12 rounded-xl border border-transparent shadow-md transition-all duration-300 ease-in-out hover:bg-[#4a81c3] hover:shadow-lg hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-[#5A9BD5] focus:ring-offset-2"
        >
          <span className="font-semibold">Upload Photo</span>
          <ArrowDown className="w-4 h-4" />
        </a>

        <a
          href="/about"
          className="bg-[#5A9BD5] text-white inline-flex items-center gap-2 px-6 h-12 rounded-xl border border-transparent shadow-md transition-all duration-300 ease-in-out hover:bg-[#4a81c3] hover:shadow-lg hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-[#5A9BD5] focus:ring-offset-2"
        >
          <span className="font-semibold">Learn More</span>
          <ArrowUpRight className="w-4 h-4" />
        </a>
      </div>
    </section>
  );
};
