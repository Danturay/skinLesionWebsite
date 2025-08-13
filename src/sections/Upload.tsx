import React, { useState, ChangeEvent } from "react";

export const UploadSection = () => {
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setPreview(url);
    }
  };

  return (
    <section className="h-screen flex flex-col justify-center items-center px-6">
      <h2 className="text-5xl font-semibold mb-10 text-[#5A9BD5] text-center max-w-4xl">
        Upload Your Skin Lesion Photo
      </h2>

      <div className="max-w-3xl mb-10 text-gray-700 text-base leading-relaxed px-4">
        <h3 className="text-2xl font-semibold mb-4 text-[#5A9BD5]">How to Take Your Photo</h3>
        <ol className="list-decimal list-inside space-y-2">
          <li>Ensure your lesion is clean and dry before taking a photo.</li>
          <li>Use natural or bright, even lighting — avoid shadows and glare.</li>
          <li>Hold your camera steady, about 10-15 cm (4-6 inches) from the lesion.</li>
          <li>Make sure the lesion fills most of the frame, but keep some surrounding skin visible for context.</li>
          <li>Focus clearly on the lesion to capture all details.</li>
        </ol>
      </div>

      <label
        htmlFor="file-upload"
        className="flex flex-col items-center justify-center w-full max-w-4xl h-96 border-4 border-dashed border-[#5A9BD5] rounded-xl cursor-pointer hover:bg-[#e6f0ff] transition duration-300"
      >
        {!preview ? (
          <>
            <svg
              className="w-20 h-20 mb-6 text-[#5A9BD5]"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 16l-4-4m0 0l4-4m-4 4h18"></path>
            </svg>
            <span className="text-[#5A9BD5] font-semibold text-2xl">
              Click or drag & drop to upload
            </span>
            <span className="text-gray-600 text-base mt-2">
              Supported formats: PNG, JPG (max 5MB)
            </span>
          </>
        ) : (
          <img
            src={preview}
            alt="Uploaded lesion preview"
            className="max-h-72 object-contain rounded-md"
          />
        )}
        <input
          id="file-upload"
          type="file"
          accept="image/png, image/jpeg"
          className="hidden"
          onChange={handleFileChange}
        />
      </label>

      <p className="mt-12 max-w-3xl text-center text-gray-700 text-sm leading-relaxed px-4">
        <strong>Disclaimer:</strong> This service is intended for informational purposes only and is not a medical device. It does <em>not</em> provide a clinical diagnosis. If you notice any of the following symptoms—such as a new or changing mole, irregular borders, multiple colors, itching, bleeding, or rapid growth—you should seek immediate evaluation from a qualified healthcare professional.
      </p>
    </section>
  );
};
