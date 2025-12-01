class Embedder:
    """Provider-agnostic embedder. Use `provider='hf'` for sentence-transformers
    or `provider='openai'` for OpenAI embeddings. Imports are lazy to avoid
    heavy startup costs.
    """

    def __init__(self, hf_model_name: str = "all-MiniLM-L6-v2"):
        self.hf_model_name = hf_model_name
        self._hf_model = None

    def _load_hf(self):
        if self._hf_model is None:
            from sentence_transformers import SentenceTransformer

            self._hf_model = SentenceTransformer(self.hf_model_name)

    def embed(self, texts, provider: str = "hf"):
        if provider == "hf":
            self._load_hf()
            return self._hf_model.encode(texts, convert_to_numpy=True)
        elif provider == "openai":
            import os
            import numpy as np
            import openai

            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set in environment")
            openai.api_key = key
            # openai.Embedding.create accepts list or single string
            resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
            return np.array([r["embedding"] for r in resp["data"]])
        else:
            raise ValueError("provider must be 'hf' or 'openai'")
