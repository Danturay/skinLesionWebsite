"use client";

import Navbar from "@/sections/Navbar";

export default function AboutPage() {
  return (
    <>
    
        <Navbar />
        <main
        className="flex min-h-screen flex-col items-center justify-center px-6 py-16 pt-32"
        style={{ backgroundColor: "#E6F0F3" }}
        >
        <div
            className="max-w-3xl rounded-2xl p-10 border shadow-lg"
            style={{
            backgroundColor: "white",
            borderColor: "#A8D5BA",
            boxShadow: "0 4px 12px rgba(90, 155, 213, 0.2)",
            }}
        >
            <h1
            className="text-4xl font-bold mb-8 text-center"
            style={{ color: "#333333" }}
            >
            About Our AI Skin Check Platform
            </h1>

            <section className="mb-10">
            <h2
                className="text-2xl font-semibold mb-4"
                style={{ color: "#5A9BD5" }}
            >
                How Our AI Model Works
            </h2>
            <p style={{ color: "#333333", lineHeight: "1.6" }}>
                Our AI model is trained using the ISIC (International Skin Imaging Collaboration) dataset, one of the most comprehensive public repositories of dermoscopic images. It leverages advanced machine learning techniques to analyze images of skin lesions and classify them into different categories, helping to identify potential skin conditions early.
            </p>
            </section>

            <section className="mb-10">
            <h2
                className="text-2xl font-semibold mb-6"
                style={{ color: "#5A9BD5" }}
            >
                Skin Conditions Detected
            </h2>
            <div style={{ color: "#333333", lineHeight: "1.6" }}>
                <p className="mb-3">
                The AI is capable of detecting a variety of common and dangerous skin conditions, including:
                </p>
                <dl className="space-y-5">
                <div>
                    <dt className="font-semibold text-teal-700">Melanoma</dt>
                    <dd>
                    A serious form of skin cancer originating in pigment-producing melanocytes, known for rapid growth and high risk of spreading.
                    </dd>
                </div>
                <div>
                    <dt className="font-semibold text-teal-700">Basal Cell Carcinoma</dt>
                    <dd>
                    The most common type of skin cancer, arising from the basal cells in the skin’s lowest layer, typically slow-growing and rarely spreads.
                    </dd>
                </div>
                <div>
                    <dt className="font-semibold text-teal-700">Actinic Keratosis</dt>
                    <dd>
                    Rough, scaly patches caused by prolonged sun exposure that can sometimes develop into squamous cell carcinoma if untreated.
                    </dd>
                </div>
                <div>
                    <dt className="font-semibold text-teal-700">Benign Keratosis</dt>
                    <dd>
                    Non-cancerous skin growths, such as seborrheic keratoses, which appear as waxy or wart-like lesions.
                    </dd>
                </div>
                <div>
                    <dt className="font-semibold text-teal-700">Benign Nevus (Moles)</dt>
                    <dd>
                    Common pigmented skin growths made up of clusters of melanocytes, usually harmless but can sometimes change and require monitoring.
                    </dd>
                </div>
                <div>
                    <dt className="font-semibold text-teal-700">Dermatofibroma</dt>
                    <dd>
                    Small, firm, benign skin nodules usually caused by minor skin injuries or insect bites.
                    </dd>
                </div>
                <div>
                    <dt className="font-semibold text-teal-700">Vascular Lesions</dt>
                    <dd>
                    Abnormalities in blood vessels visible on the skin, such as spider veins or hemangiomas.
                    </dd>
                </div>
                </dl>
            </div>
            </section>

            <section>
            <h2
                className="text-2xl font-semibold mb-4"
                style={{ color: "#5A9BD5" }}
            >
                Understanding Melanoma: Risks & Symptoms
            </h2>
            <p className="mb-4 leading-relaxed" style={{ color: "#333333" }}>
                Melanoma is the most serious form of skin cancer and can be life-threatening if not detected early. It develops when melanocytes, the cells that produce pigment in your skin, begin to grow uncontrollably.
            </p>
            <p className="mb-4 leading-relaxed" style={{ color: "#333333" }}>
                <strong style={{ color: "#F6B6A9" }}>Why early detection matters:</strong> Melanoma can spread quickly to other parts of the body, making early diagnosis crucial to successful treatment and improved survival rates.
            </p>
            <p className="mb-4 leading-relaxed" style={{ color: "#333333" }}>
                <strong style={{ color: "#F6B6A9" }}>Common symptoms to watch for:</strong>
            </p>
            <ul className="list-disc list-inside space-y-1" style={{ color: "#333333" }}>
                <li>Asymmetry: One half of the mole doesn’t match the other half.</li>
                <li>Border: Edges are irregular, ragged, or blurred.</li>
                <li>Color: Varied shades of brown, black, or even patches of pink, red, white, or blue.</li>
                <li>Diameter: Larger than 6mm (about the size of a pencil eraser), but can be smaller.</li>
                <li>Evolution: Changes in size, shape, color, or new symptoms like bleeding or itching.</li>
            </ul>
            <p className="mt-6" style={{ color: "#333333" }}>
                Our AI model aims to assist by highlighting suspicious lesions early, encouraging you to seek professional medical advice if needed.
            </p>
            </section>
        </div>
        </main>
    </>
  );
}
