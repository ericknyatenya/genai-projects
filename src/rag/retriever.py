class Retriever:
    """Minimal in-memory retriever using cosine similarity over numpy arrays.

    This is intentionally light-weight and avoids importing faiss/chroma at
    module import time. If you want an optimized backend, extend this class.
    """

    def __init__(self):
        self.embeddings = None
        self.docs = []

    def add(self, docs, vectors):
        """Add documents and their vectors (numpy arrays).

        Args:
            docs: list of strings
            vectors: 2D numpy array (n_docs x dim)
        """
        import numpy as np

        if self.embeddings is None:
            self.embeddings = np.array(vectors)
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        self.docs.extend(docs)

    def retrieve(self, query_vector, top_k: int = 5):
        import numpy as np

        if self.embeddings is None or len(self.docs) == 0:
            return []
        # cosine similarity
        q = query_vector
        emb = self.embeddings
        norms = np.linalg.norm(emb, axis=1) * (np.linalg.norm(q) + 1e-12)
        sims = (emb @ q) / norms
        idx = sims.argsort()[::-1][:top_k]
        return [(int(i), float(sims[i]), self.docs[i]) for i in idx]
