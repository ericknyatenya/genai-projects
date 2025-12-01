def generate(prompt: str, model_name: str = "gpt2", max_length: int = 128):
    """Generate text using Hugging Face `transformers` pipeline.

    This is a lightweight wrapper — install `transformers` if you want to
    use it. We import lazily to avoid heavy deps at import time.
    """
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers is not installed or failed to import") from exc
    gen = pipeline("text-generation", model=model_name)
    out = gen(prompt, max_length=max_length, do_sample=False)
    return out[0]["generated_text"]
