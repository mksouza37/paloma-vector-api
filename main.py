from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

# Load index
vectorstore = FAISS.load_local("paloma_vector_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

app = FastAPI()

class Result(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    query: str
    results: List[Result]

@app.get("/query", response_model=QueryResponse)
def query_vector(q: str = Query(..., description="User question")):
    results = vectorstore.similarity_search(q, k=3)
    return {
        "query": q,
        "results": [{"content": r.page_content, "metadata": r.metadata} for r in results]
    }
