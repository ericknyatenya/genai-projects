def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """A simple word-based chunker with overlap.

    Args:
        text: input string
        chunk_size: number of words per chunk
        overlap: number of words to overlap between chunks
    Returns:
        List[str] of text chunks
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    words = text.split()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += step
    return chunks
