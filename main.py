from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from huggingface_hub import login
from datasets import load_dataset
import json
import os

app = FastAPI()

token = os.getenv('HF_TOKEN')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
login(token=token)
# Инициализация
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
index = faiss.read_index("wikivoyage.index")
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Загрузить датасет
dataset = load_dataset("bigscience-data/roots_en_wikivoyage", split="train")

# Маленький векторайзер (например, MiniLM)
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Получить тексты
texts = dataset["text"][:10000]

class QueryRequest(BaseModel):
    text: str

@app.post("/answer")
async def get_answer(request: QueryRequest):
    query = request.text
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k=5)
    
    # Загрузить тексты (нужно сохранить их отдельно)
    rag_result = [texts[i] for i in indices[0]]
    
    system_prompt = """
You are an expert assistant specialized in providing accurate, well-researched answers based on provided context. Your role is to carefully analyze the given information and formulate clear, coherent responses that directly address the user's question.

Instructions:
1. Read the question carefully
2. Read the context provided below carefully
3. Find all information regarged to cultural characteristics or manners and customs or attractions and sights
4. Identify the most relevant information that relates to the question
5. Provide a comprehensive answer that is directly supported by the context
6. If the context does not contain sufficient information to answer the question, clearly state this
7. Avoid making assumptions or providing information not found in the context
8. Structure your answer clearly with proper formatting if needed
9. Be concise but thorough in your explanation
10. Your answer should be less than 700 symbols
"""
    user_prompt = f"Context:\n {rag_result}\nQuestion: {query}\nAnswer:"
    
    resp = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    
    return {"answer": json.loads(resp.choices[0].message.content)}