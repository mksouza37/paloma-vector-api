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
    # Use similarity with score
    docs_with_scores = vectorstore.similarity_search_with_score(q, k=5)

    # Set threshold (you can tune this — values typically between 0.6 and 0.8)
    threshold = 0.65
    filtered = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc, score in docs_with_scores
        if score >= threshold
    ]

    return {
        "query": q,
        "results": filtered
    }
