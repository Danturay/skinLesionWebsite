"use client";

import { useState } from "react";
import Navbar from "@/sections/Navbar"; // Adjust path if needed
import Image from "next/image";

export default function TrackingPage() {
  const [images, setImages] = useState<File[]>([]);
  const [error, setError] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError("");
    if (!e.target.files) return;

    const selectedFiles = Array.from(e.target.files);

    // Optional: Limit max files to 5 for example
    if (images.length + selectedFiles.length > 5) {
      setError("You can upload up to 5 images only.");
      return;
    }

    setImages((prev) => [...prev, ...selectedFiles]);
  };

  const removeImage = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <>
      <Navbar />
      <main
        className="min-h-screen px-6 py-12 pt-32"
        style={{ backgroundColor: "#E6F0F3", color: "#333333" }}
      >
        <section className="max-w-3xl mx-auto">
          <h1 className="text-4xl font-bold mb-6" style={{ color: "#5A9BD5" }}>
            Track Your Lesions Over Time
          </h1>
          <p className="mb-8 text-lg" style={{ color: "#333333" }}>
            Monitoring changes in your skin lesions is crucial for early detection
            of skin cancer and other conditions. Regularly tracking suspicious
            moles or spots helps you and your healthcare provider notice any growth,
            color change, or other signs that might need medical attention.
          </p>

          <p className="mb-6" style={{ color: "#666666" }}>
            Upload clear images of your lesion(s) taken at different times. This
            will help you compare and keep an eye on any changes.
          </p>

          {error && (
            <div
              className="mb-4 text-center font-semibold"
              style={{ color: "#F6B6A9" }}
            >
              {error}
            </div>
          )}

          <label
            htmlFor="lesion-upload"
            className="inline-block mb-4 cursor-pointer rounded-md px-6 py-3 font-semibold shadow-sm"
            style={{
              backgroundColor: "#5A9BD5",
              color: "white",
              userSelect: "none",
            }}
          >
            Select Images
          </label>
          <input
            id="lesion-upload"
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileChange}
            className="hidden"
          />

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
            {images.map((file, index) => (
              <div
                key={index}
                className="relative rounded-lg overflow-hidden border"
                style={{ borderColor: "#A8D5BA" }}
              >
                <Image
                  src={URL.createObjectURL(file)}
                  alt={`Lesion ${index + 1}`}
                  className="object-cover w-full h-32"
                />
                <button
                  onClick={() => removeImage(index)}
                  className="absolute top-1 right-1 bg-[#F6B6A9] rounded-full w-6 h-6 flex items-center justify-center text-white font-bold hover:bg-[#f59c91]"
                  aria-label="Remove image"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </section>
      </main>
    </>
  );
}
