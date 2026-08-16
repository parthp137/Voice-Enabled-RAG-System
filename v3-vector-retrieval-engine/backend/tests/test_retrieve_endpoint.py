import io, sys, json, httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 1. Health check
r = httpx.get("http://127.0.0.1:8005/health")
print("Health:", r.json())

# 2. Ensemble retrieve
r2 = httpx.post(
    "http://127.0.0.1:8005/debug/retrieve",
    json={"query": "corporation definition", "strategy": "ensemble", "top_k": 3},
)
d = r2.json()
print("Status:", r2.status_code)
print("Count:", d["count"])
for c in d["results"]:
    print(f"  {c['chunk_id']} score={c['score']:.4f} strategy={c['metadata']['strategy']} text={c['text'][:80]}...")

# 3. Single strategy retrieve
r3 = httpx.post(
    "http://127.0.0.1:8005/debug/retrieve",
    json={"query": "corporation definition", "strategy": "sentence_semantic", "top_k": 3},
)
d3 = r3.json()
print("\nSingle strategy (sentence_semantic):")
print("Count:", d3["count"])
for c in d3["results"]:
    print(f"  {c['chunk_id']} score={c['score']:.4f} text={c['text'][:80]}...")
