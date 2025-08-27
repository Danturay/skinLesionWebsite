"use client";
import Link from "next/link";
import React, { useState } from "react";
import NavLink from "./NavLink";
import { Bars3Icon, XMarkIcon } from "@heroicons/react/24/outline";
import MenuOverlay from "@/sections/MenuOverlay";
import { auth } from "@/app/firebase/config";
import { useAuthState } from "react-firebase-hooks/auth";
import { signOut } from "firebase/auth";

const navLinks = [
  { title: "Home", path: "/#home" },
  { title: "About", path: "/about" },
  { title: "Upload", path: "/#upload" },
  { title: "Tracking", path: "/tracking" },
];

const Navbar = () => {
  const [navbarOpen, setNavbarOpen] = useState(false);
  const [user] = useAuthState(auth);

  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (error) {
      console.error("Logout error:", error);
    }
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-20 bg-white bg-opacity-80 backdrop-blur-sm shadow-md">
      <div className="container mx-auto flex flex-wrap items-center justify-between px-6 py-4">
        {/* Logo */}
        <Link href="/" className="flex items-center space-x-3">
          <span className="hidden md:inline text-[#5A9BD5] font-semibold text-xl">
            SkinSafe
          </span>
        </Link>

        {/* Mobile menu toggle */}
        <div className="md:hidden">
          <button
            onClick={() => setNavbarOpen(!navbarOpen)}
            aria-label="Toggle menu"
            className="text-[#5A9BD5] hover:text-[#4178BE] focus:outline-none focus:ring-2 focus:ring-[#5A9BD5] rounded"
          >
            {!navbarOpen ? (
              <Bars3Icon className="h-6 w-6" />
            ) : (
              <XMarkIcon className="h-6 w-6" />
            )}
          </button>
        </div>

        {/* Desktop nav links */}
        <div className="hidden md:flex md:items-center md:space-x-8">
          <ul className="flex space-x-8 text-gray-700 font-medium">
            {navLinks.map(({ title, path }, i) => (
              <li key={i}>
                <NavLink href={path} title={title} className="hover:text-[#5A9BD5]" />
              </li>
            ))}
          </ul>

          {/* Right side */}
          <div className="ml-8 flex space-x-4 items-center">
            {user ? (
              <>
                <span className="text-[#5A9BD5] font-semibold">
                  Hi, {user.displayName || "User"}
                </span>
                <button
                  onClick={handleLogout}
                  className="px-4 py-2 border border-[#5A9BD5] rounded-lg text-[#5A9BD5] font-semibold hover:bg-[#5A9BD5] hover:text-white transition"
                >
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link
                  href="/login"
                  className="px-4 py-2 border border-[#5A9BD5] rounded-lg text-[#5A9BD5] font-semibold hover:bg-[#5A9BD5] hover:text-white transition"
                >
                  Login
                </Link>
                <Link
                  href="/register"
                  className="px-4 py-2 bg-[#5A9BD5] rounded-lg text-white font-semibold hover:bg-[#4178BE] transition"
                >
                  Register
                </Link>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Mobile menu overlay */}
      {navbarOpen && <MenuOverlay links={navLinks} onLinkClick={() => setNavbarOpen(false)} />}
    </nav>
  );
};

export default Navbar;
