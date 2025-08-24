"use client";

import { useEffect, useState } from "react";
import Navbar from "@/sections/Navbar";
import Image from "next/image";
import { auth, db, storage } from "@/app/firebase/config";
import { useAuthState } from "react-firebase-hooks/auth";
import {
  collection,
  addDoc,
  doc,
  getDocs,
  query,
  orderBy,
  serverTimestamp,
} from "firebase/firestore";
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";

type LesionImage = {
  id: string;
  url: string;
  createdAt: Date;
};

type Lesion = {
  id: string;
  name: string;
  images: LesionImage[];
};

export default function TrackingPage() {
  const [user, loading] = useAuthState(auth);
  const [lesions, setLesions] = useState<Lesion[]>([]);
  const [activeLesionId, setActiveLesionId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  // Redirect if not logged in
  useEffect(() => {
    if (!loading && !user) {
      window.location.href = "/login"; // adjust route
    }
  }, [loading, user]);

  // Fetch lesions for this user
  useEffect(() => {
    if (!user) return;

    const fetchLesions = async () => {
      const lesionsRef = collection(db, "users", user.uid, "lesions");
      const lesionDocs = await getDocs(lesionsRef);

      const lesionData: Lesion[] = [];
      for (const lesionDoc of lesionDocs.docs) {
        const lesionId = lesionDoc.id;
        const imagesRef = collection(
          db,
          "users",
          user.uid,
          "lesions",
          lesionId,
          "images"
        );
        const q = query(imagesRef, orderBy("createdAt", "desc"));
        const imageDocs = await getDocs(q);

        const images: LesionImage[] = imageDocs.docs.map((img) => ({
          id: img.id,
          url: img.data().url,
          createdAt: img.data().createdAt?.toDate?.() || new Date(),
        }));

        lesionData.push({
          id: lesionId,
          name: lesionDoc.data().name || "Unnamed Lesion",
          images,
        });
      }

      setLesions(lesionData);
    };

    fetchLesions();
  }, [user]);

  // Create a new lesion
  const createLesion = async () => {
    if (!user) return;
    const lesionRef = await addDoc(collection(db, "users", user.uid, "lesions"), {
      name: `Lesion ${lesions.length + 1}`,
      createdAt: serverTimestamp(),
    });
    setLesions((prev) => [
      ...prev,
      { id: lesionRef.id, name: `Lesion ${lesions.length + 1}`, images: [] },
    ]);
    setActiveLesionId(lesionRef.id);
  };

  // Upload images to a lesion
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!user || !activeLesionId) return;
    if (!e.target.files) return;

    const files = Array.from(e.target.files);
    setUploading(true);

    for (const file of files) {
      const storageRef = ref(
        storage,
        `users/${user.uid}/lesions/${activeLesionId}/${file.name}`
      );
      await uploadBytes(storageRef, file);
      const url = await getDownloadURL(storageRef);

      const imgRef = await addDoc(
        collection(db, "users", user.uid, "lesions", activeLesionId, "images"),
        {
          url,
          createdAt: serverTimestamp(),
        }
      );

      setLesions((prev) =>
        prev.map((lesion) =>
          lesion.id === activeLesionId
            ? {
                ...lesion,
                images: [
                  { id: imgRef.id, url, createdAt: new Date() },
                  ...lesion.images,
                ],
              }
            : lesion
        )
      );
    }

    setUploading(false);
  };

  const daysAgo = (date: Date) => {
    const diff = Math.floor((Date.now() - date.getTime()) / (1000 * 60 * 60 * 24));
    return diff === 0 ? "Today" : `${diff} day(s) ago`;
  };

  if (loading || !user) {
    return <div className="text-center mt-20">Loading...</div>;
  }

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

          {/* Add new lesion */}
          <button
            onClick={createLesion}
            className="mb-6 bg-[#5A9BD5] text-white px-6 py-3 rounded-lg shadow hover:bg-[#4a89c9] transition"
          >
            + Add New Lesion
          </button>

          {/* Lesions list */}
          {lesions.map((lesion) => (
            <div key={lesion.id} className="mb-10 p-6 bg-white rounded-lg shadow">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-semibold text-[#5A9BD5]">
                  {lesion.name}
                </h2>
                <label className="cursor-pointer bg-[#A8D5BA] text-white px-4 py-2 rounded-md hover:bg-[#8cc9a7]">
                  {uploading && activeLesionId === lesion.id
                    ? "Uploading..."
                    : "Add Image"}
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={(e) => {
                      setActiveLesionId(lesion.id);
                      handleFileChange(e);
                    }}
                  />
                </label>
              </div>

              {/* Images */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {lesion.images.map((img) => (
                  <div
                    key={img.id}
                    className="relative rounded-lg overflow-hidden border"
                    style={{ borderColor: "#A8D5BA" }}
                  >
                    <Image
                      src={img.url}
                      alt="Lesion"
                      width={400}
                      height={400}
                      className="object-cover w-full h-32"
                    />
                    <div className="absolute bottom-1 left-1 text-xs bg-white px-2 py-1 rounded">
                      {daysAgo(img.createdAt)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </section>
      </main>
    </>
  );
}
