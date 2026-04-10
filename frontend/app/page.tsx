"use client";

import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult("");
    }
  };

  const handleSubmit = async () => {
    if (!file) return alert("Upload MRI image");

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data.prediction || "No result");
    } catch {
      setResult("API Error");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-gray-800 text-white flex flex-col">

      {/* Header */}
      <header className="flex justify-between items-center px-10 py-6">
        <h1 className="text-xl font-bold tracking-wide">
          🧠 Alzheimer AI
        </h1>
        <p className="text-gray-400 text-sm">
          MRI Classification System
        </p>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 items-center justify-center gap-16 px-10">

        {/* Left Side */}
        <div className="max-w-md">
          <h2 className="text-4xl font-bold leading-tight">
            Detect Alzheimer’s <br /> using AI
          </h2>
          <p className="text-gray-400 mt-4">
            Upload MRI scans and get instant classification powered by EfficientNet.
          </p>

          {/* Upload */}
          <label className="mt-6 block cursor-pointer border border-gray-600 rounded-xl p-6 text-center hover:border-indigo-400 transition">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
            <p className="text-gray-400">Click to upload MRI image</p>
          </label>

          {/* Button */}
          <button
            onClick={handleSubmit}
            className="mt-5 px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl hover:scale-105 transition transform"
          >
            {loading ? "Analyzing..." : "Run Detection"}
          </button>

          {/* Result */}
          {result && (
            <div className="mt-6">
              <p className="text-gray-400 text-sm">Prediction</p>
              <h3 className="text-2xl font-bold text-green-400">
                {result}
              </h3>
            </div>
          )}
        </div>

        {/* Right Side Preview */}
        {preview && (
          <div className="w-[350px] h-[350px]">
            <img
              src={preview}
              alt="preview"
              className="rounded-2xl w-full h-full object-cover border border-gray-700 shadow-xl"
            />
          </div>
        )}
      </div>
    </div>
  );
}