# genai-projects

Scaffold for retrieval-augmented generation (RAG) and LLM experiments.

Contents
- `src/rag/` — chunker, embedder, retriever (lightweight, provider-agnostic)
- `src/llm/` — wrappers for OpenAI and Hugging Face generation
- `api/rag_api.py` — simple FastAPI endpoint (stubbed)
- `ui/streamlit_app.py` — simple Streamlit demo app
- `notebooks/` — example notebooks (placeholders)

Quickstart (dev)

1. Create a venv and install minimal deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start the API:

```bash
uvicorn api.rag_api:app --reload --port 8000
```

3. Run the Streamlit UI (in a *separate* terminal):

```bash
streamlit run ui/streamlit_app.py
```

Notes
- The scaffold includes both OpenAI and sentence-transformers usage examples. Set `OPENAI_API_KEY` in your environment to use the OpenAI wrapper.
- The modules use lazy imports to avoid heavy install-time costs. See `src/rag/embedder.py` for examples.

License: MIT (see `LICENSE`)
