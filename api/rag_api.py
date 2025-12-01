from fastapi import FastAPI
from pydantic import BaseModel

class Query(BaseModel):
    query: str
    provider: str = "hf"

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rag")
def rag(q: Query):
    # This is a lightweight stub demonstrating shape of request/response.
    # A real implementation would: chunk -> embed -> retrieve -> generate
    return {"answer": f"(stub) received query: {q.query}", "provider": q.provider}
