import os
import asyncio
import requests
import json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

import numpy as np
from together import Together
from fastembed.embedding import TextEmbedding
from flashrank import Ranker, RerankRequest
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Konfigurazio-parametroak
load_dotenv()
DATA_DIR = Path("data")
DB_DIR = Path("./db")
PDF_FILE = "NVIDIAAn.pdf"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

# API gakoak
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

class DocumentProcessor:
    @staticmethod
    async def parse_pdf(file_path: str) -> str:
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            max_timeout=5000,
            content_guideline_instruction="Extract financial tables and relevant text keeping the structure intact."
        )
        documents = await parser.aload_data(file_path)
        full_context = "\n".join(doc.text for doc in documents)
        return full_context

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
            if start < 0:
                start = 0
        return chunks

class VectorDatabase:
    def __init__(self):
        self.client = QdrantClient(path=str(DB_DIR))
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        self.ranker = Ranker(model_name=RERANK_MODEL)
       
    def initialize_collection(self, collection_name: str):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,  # Embedding tamaina
                distance=Distance.COSINE
            )
        )
   
    def add_documents(self, collection_name: str, chunks: List[str]):
        embeddings = list(self.embedder.embed(chunks))
       
        points = [
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"text": chunk}
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
       
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

class QAEngine:
    def __init__(self):
        # Together APIa erabiltzeko bezeroa inizializatu (API gakoa ingurune-aldagaietatik jasotzen da)
        self.client = Together()

    def generate_response(self, context: str, question: str) -> str:
        """
        Finantza-testuinguruan eta galderan oinarritutako erantzuna sortzen du,
        Together-en txat complementions endpoint-a erabiliz streaming bidez.
        """
        prompt = f"""Analyze the following financial context and answer the question:
        
Context:
{context}

Question: {question}

Answer (be precise with figures and dates, mention the source):
"""
        messages = [{"role": "user", "content": prompt}]

        # Together-en endpoint-erako deia, streaming-a aktibatuta
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=messages,
            max_tokens=2056,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<｜end▁of▁sentence｜>"],
            stream=True
        )

        full_response = ""
        # Jasotako token bakoitza ikusi eta denbora errealean erakutsi
        for token in response:
            if hasattr(token, 'choices'):
                content = token.choices[0].delta.content
                full_response += content
                print(content, end='', flush=True)
        print()  # Streaming-a amaitzean, lerro-jauzia

        return full_response

async def main():
    # 1. urratsa: PDF prozesatu
    processor = DocumentProcessor()
    parsed_text = await processor.parse_pdf(PDF_FILE)
   
    # 2. urratsa: Testua gorde eta zatitu
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "parsed.md").write_text(parsed_text)
    chunks = processor.chunk_text(parsed_text)
   
    # 3. urratsa: Hasieratu datu-base bektoriala eta dokumentuak gehitu
    db = VectorDatabase()
    db.initialize_collection("financial_reports")
    db.add_documents("financial_reports", chunks)
   
    # 4. urratsa: Bilatu eta erantzuna sortu
    query = "¿Cúal es el pronostico o la previsión que harías con NVIDIA para el resto del año 2024? ¿En qué porcentaje subirán las acciones?"
   
    # Bilaketa semantikoa
    embeddings = list(db.embedder.embed([query]))
    query_vector = embeddings[0].tolist()

    search_results = db.client.search(
        collection_name="financial_reports",
        query_vector=query_vector,
        limit=10
    )
   
    # Re-ranker-a erabili
    ranked_results = db.ranker.rerank(RerankRequest(
        query=query,
        passages=[{"text": hit.payload["text"]} for hit in search_results]
    ))
   
    # Testuingurua erabili
    context = "\n".join([doc["text"] for doc in ranked_results])
   
    # Erantzuna sortu
    qa = QAEngine()
    qa.generate_response(context, query)

if __name__ == "__main__":
    asyncio.run(main())