import numpy as np
from src.rag.chunker import chunk_text
from src.rag.retriever import Retriever


def test_chunker_basic():
    text = "This is a short document used for testing chunker behavior."
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    # each chunk should be a substring of the original
    for c in chunks:
        assert isinstance(c, str)
        assert c in text or c in text.replace(" ", " ")


def test_retriever_basic():
    docs = ["apple banana", "orange pear", "banana fruit"]
    # build simple vectors (small dimensionality) such that dot product favors doc 0
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1]])
    retr = Retriever()
    retr.add(docs, vecs)

    # query vector similar to doc 0
    q = np.array([1.0, 0.0])
    results = retr.retrieve(q, top_k=2)
    assert len(results) == 2
    # first returned doc should be index 0
    first_idx, first_score, first_doc = results[0]
    assert first_idx == 0
    assert first_doc == docs[0]
