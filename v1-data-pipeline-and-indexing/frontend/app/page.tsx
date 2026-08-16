export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-slate-950 text-slate-100">
      <div className="max-w-3xl text-center space-y-6">
        <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl bg-gradient-to-r from-blue-400 via-teal-300 to-indigo-400 bg-clip-text text-transparent">
          HHG Voice RAG
        </h1>
        <p className="text-lg text-slate-400">
          Multilingual Indic Voice Search & Question Answering powered by offline FAISS index & E5 Embeddings.
        </p>
        <div className="inline-flex items-center px-4 py-2 rounded-full border border-teal-500/30 bg-teal-500/10 text-teal-300 text-sm font-medium">
          Backend Pipeline Initialized
        </div>
      </div>
    </main>
  );
}
