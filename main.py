from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

# Load index
vectorstore = FAISS.load_local("paloma_vector_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

app = FastAPI()

class Result(BaseModel):
    content: str
    metadata: dict

@app.get("/query", response_model=List[Result])
def query_vector(q: str = Query(..., description="User question")):
    results = vectorstore.similarity_search(q, k=3)
    return [{"content": r.page_content, "metadata": r.metadata} for r in results]
