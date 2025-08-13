"use client";

export const SectionHeader = ({
    title,
    eyebrow,
    description,
}: {
    title: string;
    eyebrow: string;
    description: string;
} ) => {
    return (
        <>
            <div className="flex justify-center">
                <p className="uppercase font-semibold tracking-widest bg-gradient-to-r from-[#F3DFA7] to-[#C4A045] text-center text-transparent bg-clip-text">{eyebrow}</p>
            </div>
                <h2 className="font-serif text-3xl md:text-5xl text-center mt-5 ">{title}</h2>
                <p className="text-center md:text-lg lg:text-xl text-white/60 mt-4 max-w-md mx-auto">{description}</p>
            <div className="mt-10 md:mt-20 flex flex-col gap-20"></div>
        
        </>

    )
}