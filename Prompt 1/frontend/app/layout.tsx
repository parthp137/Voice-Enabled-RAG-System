import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HHG Voice RAG - Multilingual Voice AI",
  description: "Offline-indexed Multilingual Voice RAG System built with Next.js, FastAPI, FAISS, and Indic models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-slate-950 text-slate-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}
