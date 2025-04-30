import os
import re
import logging
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel

from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
from llama_parse import LlamaParse
from together import Together

# Configuración
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# MongoDB client
load_dotenv()
_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
_db = _client[MONGODB_DB]

class DocumentAgent:
    def __init__(self):
        self.llm_client = Together(api_key=TOGETHER_API_KEY)
        self.model_name = LLM_MODEL
        self.company_list = [c['name'] for c in _db.companies.find({}, {'name': 1})]

    def select_financial_entity(self, filename: str, first_page_content: str) -> str:
        prompt = f"""
        You are a document classification agent.
        Your task is to determine whether a given document is clearly associated with a specific entity.

        You will receive:
        - The file name of the document.
        - The text content of the document's first page.

        Your instructions:
        - Analyze both the file name and the first page content.
        - Identify any direct or indirect references to identifiable entities such as:
        - Companies (e.g., Google, Amazon)
        - Cryptocurrencies (e.g., Bitcoin, Ethereum)
        - ETFs (e.g., SPY, QQQ)
        - Investment funds or financial instruments

        - If the document clearly belongs to one of these entities, output the entity name.
        - If there is no clear match, output "No match found".

        Be strict: only confirm a match if there is strong evidence (like the entity name, ticker symbol, related brand, or unique identifiers).

        Format your answer exactly like this, do NOT add anything else including explanations:

        Company: [Entity Name or "No match found"]

        ----

        File name: {filename}
        First page content: {first_page_content}
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
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
        for token in response:
            if hasattr(token, 'choices'):
                content = token.choices[0].delta.content
                full_response += content
        print(f"Full response: {full_response}")
        clean_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        print(f"Response: {clean_response}")
        match = re.search(r"Company:\s*(.+)", clean_response)
        if match:
            return match.group(1).strip()
        return None

class DocumentProcessor:
    @staticmethod
    async def parse_pdf(file_path: str) -> List[str]:
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            parse_mode="parse_page_with_agent",
            result_type="text"
        )
        pages = await parser.aload_data(file_path)
        return [p.text for p in pages]

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 128) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

class VectorMongoDB:
    def __init__(self, collection_name: str):
        self.coll = _db[collection_name]
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

    def create_vector_index(self, index_name: str):
        index = SearchIndexModel(
            definition={"fields": [{"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine", "quantization": "scalar"}]},
            name=index_name,
            type="vectorSearch"
        )
        try:
            self.coll.create_search_index(index)
            logging.info(f"Index '{index_name}' ready on '{self.coll.name}'")
        except Exception as e:
            logging.error(f"Error creating index: {e}")

    def drop_vector_index(self, index_name: str):
        try:
            self.coll.drop_search_index(index_name)
            logging.info(f"Dropped index '{index_name}' from '{self.coll.name}'")
        except Exception as e:
            logging.error(f"Error dropping index: {e}")

    def add_documents(self, chunks: List[dict]):
        texts = [chunk["text"] for chunk in chunks]
        embeddings = list(self.embedder.embed(texts))
        
        docs = []
        for chunk, emb in zip(chunks, embeddings):
            doc = {
                "text": chunk["text"],
                "embedding": emb.tolist(),
                "metadata": chunk.get("metadata", {})
            }
            docs.append(doc)
        
        try:
            self.coll.insert_many(docs)
            logging.info(f"Inserted {len(docs)} docs en '{self.coll.name}'")
        except Exception as e:
            logging.error(f"Error de inserción: {e}")

    def semantic_search(self, query: str, k: int = 10, num_candidates: int = 100) -> List[dict]:
        q_emb = list(self.embedder.embed([query]))[0].tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": q_emb,
                    "path": "embedding",
                    "limit": k,
                    "numCandidates": num_candidates
                }
            },
            {
                "$project": {
                    "text": 1,
                    "_id": 0
                }
            }
        ]
        try:
            return list(self.coll.aggregate(pipeline))
        except Exception as e:
            logging.error(f"Search error: {e}")
            return []

class QAEngine:
    def __init__(self):
        self.client = Together(api_key=TOGETHER_API_KEY)

    def generate_response(self, context: str, question: str) -> str:
        prompt = f"""
        Analyze the following context and answer the question:

        Context:\n{context}
        Question: {question}
        Answer:
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
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
        for token in response:
            if hasattr(token, 'choices'):
                content = token.choices[0].delta.content
                full_response += content
        return re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()