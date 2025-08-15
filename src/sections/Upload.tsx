import React, { useState, ChangeEvent } from "react";
import Image from "next/image";

export const UploadSection = () => {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const labelMap: Record<string, string> = {
    nv: "Melanocytic Nevus (benign mole)",
    mel: "Melanoma (skin cancer)",
    bkl: "Benign Keratosis",
    bcc: "Basal Cell Carcinoma",
    akiec: "Actinic Keratosis",
    vasc: "Vascular Lesion",
    df: "Dermatofibroma",
  };

  const explanations: Record<string, string> = {
    "Melanocytic Nevus (benign mole)":
      "A common and usually harmless mole caused by clusters of pigment cells. Monitoring for changes is still important.",
    "Melanoma (skin cancer)":
      "A serious type of skin cancer that can spread quickly. Early detection and treatment greatly improve outcomes.",
    "Benign Keratosis":
      "A harmless skin growth often related to aging or sun exposure. Usually requires no treatment unless irritated.",
    "Basal Cell Carcinoma":
      "A slow-growing form of skin cancer that rarely spreads but should be treated to prevent local damage.",
    "Actinic Keratosis":
      "A precancerous lesion caused by sun damage. May develop into skin cancer if untreated.",
    "Vascular Lesion":
      "Usually harmless and caused by growth or dilation of blood vessels. Rarely needs treatment unless symptomatic.",
    "Dermatofibroma":
      "A common benign skin lesion that is firm and slightly pigmented. Typically harmless.",
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      const url = URL.createObjectURL(selectedFile);
      setPreview(url);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.top3_predictions) {
        const predictionsArray = data.top3_predictions.map((p: any) => ({
          label: labelMap[p.class] || p.class,
          confidence: p.confidence,
        }));
        setResult({ predictions: predictionsArray });
      } else {
        setResult(null);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600 font-semibold";
    if (confidence >= 0.5) return "text-orange-500 font-semibold";
    return "text-red-500 font-semibold";
  };

  const renderPrediction = () => {
    if (!result || !result.predictions) return null;

    const sorted = [...result.predictions].sort(
      (a, b) => b.confidence - a.confidence
    );
    const top = sorted[0];
    const others = sorted.slice(1);

    return (
      <div className="mt-6 p-6 bg-white rounded-lg shadow max-w-2xl">
        <h3 className="text-2xl font-bold text-[#5A9BD5] mb-4">
          Analysis Results
        </h3>

        <p className="text-lg text-gray-800 mb-2">
          The most likely condition detected is{" "}
          <span className="font-semibold">{top.label}</span> with{" "}
          <span className={getConfidenceColor(top.confidence)}>
            {(top.confidence * 100).toFixed(1)}%
          </span>{" "}
          confidence.
        </p>

        {others.length > 0 && (
          <p className="text-gray-700 mb-4">
            Other possibilities include{" "}
            {others
              .map(
                (o) =>
                  `${o.label} (${(o.confidence * 100).toFixed(1)}%)`
              )
              .join(", ")}.
          </p>
        )}

        {explanations[top.label] && (
          <p className="text-gray-700 mb-4">{explanations[top.label]}</p>
        )}

        <p className="text-gray-700 italic">
          <strong>Next steps:</strong> While AI can assist in identifying skin
          lesions, it is not a substitute for a medical diagnosis. Please
          consult a qualified dermatologist for a professional evaluation,
          especially if the lesion is changing in size, shape, or color.
        </p>
      </div>
    );
  };

  return (
    <section className="h-screen flex flex-col justify-center items-center px-6" id="upload">
      <h2 className="text-5xl font-semibold mb-6 text-[#5A9BD5] text-center max-w-4xl mt-20">
        Upload Your Skin Lesion Photo
      </h2>

      {/* Instructions */}
      <div className="max-w-3xl mb-10 text-gray-700 text-base leading-relaxed px-4">
        <ol className="list-disc list-inside space-y-2 text-center">
          <li>Ensure your lesion is clean and dry before taking a photo.</li>
          <li>Use natural or bright, even lighting and avoid shadows and glare.</li>
          <li>Hold your camera steady, about 10-15 cm (4-6 inches) from the lesion.</li>
          <li>Make sure the lesion fills most of the frame, but keep some surrounding skin visible for context.</li>
          <li>Focus clearly on the lesion to capture all details.</li>
        </ol>
      </div>

      {/* Upload Box */}
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
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M7 16l-4-4m0 0l4-4m-4 4h18"
              ></path>
            </svg>
            <span className="text-[#5A9BD5] font-semibold text-2xl">
              Click or drag & drop to upload
            </span>
            <span className="text-gray-600 text-base mt-2">
              Supported formats: PNG, JPG (max 5MB)
            </span>
          </>
        ) : (
          <Image
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

      {/* Upload Button */}
      <button
        onClick={handleUpload}
        disabled={loading}
        className="mt-6 bg-[#5A9BD5] text-white px-6 py-3 rounded-lg shadow-lg hover:bg-[#4a89c9] transition duration-300"
      >
        {loading ? "Processing..." : "Analyse Image"}
      </button>

      {/* Result */}
      {renderPrediction()}

      <p className="mt-12 max-w-3xl text-center text-gray-700 text-sm leading-relaxed px-4">
        <strong>Disclaimer:</strong> This service is intended for informational purposes only and is not a medical device. It does <em>not</em> provide a clinical diagnosis. If you notice any of the following symptoms such as a new or changing mole, irregular borders, multiple colors, itching, bleeding, or rapid growth, you should seek immediate evaluation from a qualified healthcare professional.
      </p>
    </section>
  );
};
