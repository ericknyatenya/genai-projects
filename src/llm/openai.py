def generate(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
    """Generate text using OpenAI ChatCompletion (sync).

    Requires `OPENAI_API_KEY` in env.
    """
    import os
    import openai

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    openai.api_key = key
    resp = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature
    )
    return resp["choices"][0]["message"]["content"].strip()
