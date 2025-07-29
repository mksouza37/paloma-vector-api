from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("intfloat/multilingual-e5-small")

class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str = "intfloat/multilingual-e5-small"

@app.get("/embed", response_model=EmbedResponse)
def embed_text(q: str = Query(...)):
    # E5 exige prefixo "query: "
    vector = model.encode(f"query: {q}").tolist()
    return {
        "embedding": vector,
        "model": "intfloat/multilingual-e5-small"
    }
