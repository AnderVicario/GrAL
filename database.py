import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel

import numpy as np
from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
from llama_parse import LlamaParse
from together import Together

# Configuración
load_dotenv()
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

PDF_FILE = "data/NVIDIAAn.pdf"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
COLLECTION_NAME = "financial_data"

# MongoDB client
client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
db = client["data"]
collection = db[COLLECTION_NAME]

# Crear índice vectorial (una sola vez)
def create_search_index():
    search_index_model = SearchIndexModel(
        definition={
            "fields": [{
                "type": "vector",
                "path": "embedding",
                "numDimensions": 768,
                "similarity": "cosine",
                "quantization": "scalar"
            }]
        },
        name="vector_index",
        type="vectorSearch"
    )
    collection.create_search_index(search_index_model)

class DocumentProcessor:
    @staticmethod
    async def parse_pdf(file_path: str) -> str:
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            parse_mode="parse_page_with_agent",
            result_type="markdown"
        )
        documents = await parser.aload_data(file_path)
        return "\n".join(doc.text for doc in documents)

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
        return chunks

class VectorMongoDB:
    def __init__(self):
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

    def add_documents(self, chunks: List[str]):
        embeddings = self.embedder.embed(chunks)
        docs = [{
            "text": chunk,
            "embedding": [float(x) for x in embedding]
        } for chunk, embedding in zip(chunks, embeddings)]
        collection.insert_many(docs)

    def search(self, query: str, k: int = 10) -> List[str]:
        embeddings = list(self.embedder.embed([query]))
        query_vec = embeddings[0]
        query_embedding = [float(x) for x in query_vec]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": k
                }
            },
            {"$project": {"text": 1, "_id": 0}}
        ]

        cursor = collection.aggregate(pipeline)
        return [doc["text"] for doc in cursor]

class QAEngine:
    def __init__(self):
        self.client = Together()

    def generate_response(self, context: str, question: str) -> str:
        prompt = f"""Analyze the following context and answer the question:

        Context:
        {context}

        Question: {question}

        Answer (include precise figures and source if relevant):"""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=2048,
            stream=True
        )

        answer = ""
        for token in response:
            if hasattr(token, 'choices'):
                content = token.choices[0].delta.content
                answer += content
                print(content, end="", flush=True)
        print()
        return answer

async def main():
    # create_search_index()  # Ejecutar una sola vez

    processor = DocumentProcessor()
    parsed_text = await processor.parse_pdf(PDF_FILE)
    chunks = processor.chunk_text(parsed_text)

    # Subir embeddings
    vector_db = VectorMongoDB()
    vector_db.add_documents(chunks)

    # Consulta
    query = "What has been the recent revenues of gaming?"
    results = vector_db.search(query)

    # Rerankeo
    passages = [
        {"id": i, "text": txt, "metadata": {"source": "chunk-"+str(i)}}
        for i, txt in enumerate(results, start=1)
    ]
    print("Passages:", passages)
    ranker = Ranker(model_name=RERANK_MODEL)
    reranked = ranker.rerank(RerankRequest(
        query=query,
        passages=passages
    ))
    print(reranked)
    context = "\n".join([r["text"] for r in reranked])

    # Generar respuesta
    qa = QAEngine()
    qa.generate_response(context, query)

if __name__ == "__main__":
    asyncio.run(main())
