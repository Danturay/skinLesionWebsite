"use client";
import logo from "@/assets/images/logo.png";
import Image from "next/image";
import { twMerge } from 'tailwind-merge';

export const CardHeader = ({
    title,
    description,
    className,
}: {
    title: string;
    description?: string;
    className?: string;
}) => {
    return (
        <div className={twMerge("space-y-2", className)}>
            <div className="inline-flex items-center gap-2">
                <Image src={logo} alt="Logo" width={37} height={37}/>
                <h3 className="font-serif text-3xl">{title}</h3>
            </div>
            <p className="text-sm lg:text-base lg:max-w-xs text-white/60">{description}</p>
        </div>
    );
};