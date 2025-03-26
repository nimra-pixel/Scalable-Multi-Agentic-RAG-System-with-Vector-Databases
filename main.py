import os
import faiss
import openai
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from fastapi import FastAPI, HTTPException

# Initialize FastAPI
app = FastAPI()

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_faiss_index():
    """Creates a FAISS vector database."""
    embedding_model = OpenAIEmbeddings()
    texts = ["What is RAG?", "How does FAISS work?", "Explain multi-agent systems."]
    embeddings = [embedding_model.embed_query(text) for text in texts]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, texts

# Create Vector Store
faiss_index, stored_texts = create_faiss_index()

@app.get("/query")
def query_rag(input_text: str):
    """Queries the RAG system and returns a response."""
    try:
        embedding_model = OpenAIEmbeddings()
        query_vector = np.array([embedding_model.embed_query(input_text)], dtype=np.float32)
        distances, indices = faiss_index.search(query_vector, k=1)
        matched_text = stored_texts[indices[0][0]]
        response = OpenAI().call(f"{matched_text}")
        return {"query": input_text, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
