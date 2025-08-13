"use client";

import { useState } from "react";
import { auth } from "../firebase/config";
import { signInWithEmailAndPassword } from "firebase/auth";

export default function LoginPage() {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    setError("");
    setLoading(true);

    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        formData.email,
        formData.password
      );

      console.log("User logged in:", userCredential.user);

      window.location.href = "/dashboard";
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      className="flex min-h-screen items-center justify-center px-4"
      style={{ backgroundColor: "#E6F0F3" }}
    >
      <div
        className="w-full max-w-md rounded-2xl p-8 border shadow-lg"
        style={{
          backgroundColor: "white",
          borderColor: "#A8D5BA",
          boxShadow: "0 4px 12px rgba(90, 155, 213, 0.2)",
        }}
      >
        <h1
          className="text-3xl font-bold text-center mb-6"
          style={{ color: "#333333" }}
        >
          Log In to Your Account
        </h1>
        <p
          className="text-sm text-center mb-8"
          style={{ color: "#666666" /* softer than #333 */ }}
        >
          Welcome back! Please log in to access your AI skin check platform.
        </p>

        {error && (
          <div
            className="mb-4 text-sm text-center"
            style={{ color: "#F6B6A9" /* soft coral for errors */ }}
          >
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Email */}
          <div>
            <label
              className="block text-sm font-medium mb-1"
              style={{ color: "#333333" }}
            >
              Email Address
            </label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="you@example.com"
              required
              className="w-full rounded-lg px-4 py-2 focus:outline-none focus:ring-2"
              style={{
                border: "1px solid #A8D5BA",
                color: "#333333",
                backgroundColor: "white",
                boxShadow: "none",
              }}
              onFocus={e =>
                (e.currentTarget.style.boxShadow = "0 0 6px #5A9BD5")
              }
              onBlur={e => (e.currentTarget.style.boxShadow = "none")}
            />
          </div>

          {/* Password */}
          <div>
            <label
              className="block text-sm font-medium mb-1"
              style={{ color: "#333333" }}
            >
              Password
            </label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="********"
              required
              className="w-full rounded-lg px-4 py-2 focus:outline-none focus:ring-2"
              style={{
                border: "1px solid #A8D5BA",
                color: "#333333",
                backgroundColor: "white",
                boxShadow: "none",
              }}
              onFocus={e =>
                (e.currentTarget.style.boxShadow = "0 0 6px #5A9BD5")
              }
              onBlur={e => (e.currentTarget.style.boxShadow = "none")}
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full font-semibold py-2 px-4 rounded-lg shadow-sm transition disabled:opacity-50"
            style={{
              backgroundColor: loading ? "#4178BE" : "#5A9BD5",
              color: "white",
            }}
          >
            {loading ? "Logging in..." : "Log In"}
          </button>
        </form>

        <p
          className="text-sm text-center mt-6"
          style={{ color: "#333333" }}
        >
          Don't have an account?{" "}
          <a
            href="/register"
            style={{ color: "#5A9BD5", textDecoration: "underline" }}
          >
            Register
          </a>
        </p>
      </div>
    </main>
  );
}
