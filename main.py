from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Modelo leve e multilíngue (roda em 512MB)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"

@app.get("/embed", response_model=EmbedResponse)
def embed_text(q: str = Query(...)):
    vector = model.encode(q).tolist()
    return {
        "embedding": vector,
        "model": "paraphrase-multilingual-MiniLM-L12-v2"
    }
