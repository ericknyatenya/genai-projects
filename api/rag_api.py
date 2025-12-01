from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.chunker import chunk_text
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever

# --- Global state (initialized on startup) ---
_retriever: Optional[Retriever] = None
_helper = None  # either None (HF), sklearn vectorizer, or vocab dict


def embed_texts_with_fallback(texts):
    """Try HF, fallback to sklearn TF-IDF, then to simple count vectorizer."""
    try:
        emb = Embedder()
        vecs = emb.embed(texts, provider="hf")
        return np.asarray(vecs), None
    except Exception:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            v = TfidfVectorizer()
            X = v.fit_transform(texts)
            return X.toarray(), v
        except Exception:
            # simple count vectorizer
            tokens = [t.lower().split() for t in texts]
            vocab = {}
            for toklist in tokens:
                for t in toklist:
                    if t not in vocab:
                        vocab[t] = len(vocab)
            X = np.zeros((len(texts), len(vocab)), dtype=float)
            for i, toklist in enumerate(tokens):
                for t in toklist:
                    X[i, vocab[t]] += 1.0
            return X, vocab


def embed_query_with_fallback(query, helper):
    """Embed query using the same helper as documents."""
    try:
        emb = Embedder()
        vec = emb.embed([query], provider="hf")[0]
        return np.asarray(vec)
    except Exception:
        if helper is None:
            return np.zeros(1, dtype=float)
        try:
            return helper.transform([query]).toarray()[0]
        except Exception:
            vocab = helper
            vec = np.zeros(len(vocab), dtype=float)
            for t in query.lower().split():
                if t in vocab:
                    vec[vocab[t]] += 1.0
            return vec


def init_retriever():
    """Load documents, chunk, embed, and build retriever."""
    global _retriever, _helper
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_docs.txt"
    if not data_path.exists():
        raise RuntimeError(f"Data file not found: {data_path}")
    
    docs = [line.strip() for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_chunks = []
    for d in docs:
        ch = chunk_text(d, chunk_size=20, overlap=5)
        all_chunks.extend(ch)
    
    vectors, helper = embed_texts_with_fallback(all_chunks)
    _helper = helper
    
    _retriever = Retriever()
    _retriever.add(all_chunks, vectors)


class Query(BaseModel):
    query: str
    top_k: int = 3
    provider: str = "hf"


class RAGResponse(BaseModel):
    query: str
    context: List[str]
    answer: str


app = FastAPI(title="genai-projects RAG API")


@app.on_event("startup")
async def startup():
    """Initialize retriever on startup."""
    try:
        init_retriever()
    except Exception as e:
        print(f"Warning: failed to initialize retriever: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "retriever_ready": _retriever is not None}


@app.post("/rag", response_model=RAGResponse)
def rag(q: Query):
    """Run RAG: retrieve relevant docs and synthesize an answer."""
    global _retriever, _helper
    
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    # embed query
    qvec = embed_query_with_fallback(q.query, _helper)
    
    # retrieve
    results = _retriever.retrieve(qvec, top_k=q.top_k)
    context = [r[2] for r in results]
    
    # synthesize answer (join context if generation unavailable)
    try:
        from src.llm.hf import generate as hf_generate
        
        prompt = q.query + "\n\nContext:\n" + "\n\n".join(context)
        answer = hf_generate(prompt, model_name="gpt2", max_length=128)
    except Exception:
        # fallback: join context
        answer = " ".join(context)
    
    return RAGResponse(query=q.query, context=context, answer=answer)
