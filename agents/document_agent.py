import os
import re
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel
from fastembed import TextEmbedding
from together import Together

# Load environment variables and configure logging
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
_db = _client[MONGODB_DB]

class DocumentAgent:
    def __init__(self):
        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

    def select_final_analysis(self, filename, first_page_content, user_company_list):
        prompt = f"""
        You are a document classification agent.
        Your task is to determine whether a given document belongs to any company from a list provided by the user.

        You will receive:
        - The file name of the document.
        - The text content of the document's first page.
        - A list of company names.

        Your instructions:
        - Analyze both the file name and the first page content.
        - Identify any direct or indirect references to the companies in the list.
        - If the document clearly belongs to one of the companies, output the company name.
        - If there is no clear match, output "No match found".

        Be strict: only confirm a match if there is enough evidence (like the company name appearing, a related brand, or unique identifiers).

        Format your answer like this:

        Company: [Company Name]

        ----

        File name: {filename}
        First page content: {first_page_content}
        User company list: {user_company_list}

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
        clean_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        match = re.search(r"Company:\s*(.+)", clean_response)
        if match:
            return match.group(1).strip()
        return "No match found"

    def list_financial_entities(self):
        return _db.list_collection_names()

    def connect_collection(self, collection_name: str):
        if collection_name not in _db.list_collection_names():
            coll = _db.create_collection(collection_name)
            logging.info(f"Collection '{collection_name}' created.")
        else:
            coll = _db[collection_name]
            logging.info(f"Collection '{collection_name}' exists.")
        return coll

    def create_vector_index(self, collection_name: str, index_name: str = "vector_index"):
        coll = self.connect_collection(collection_name)
        index = SearchIndexModel(
            definition={
                "fields": [{
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 768,
                    "similarity": "cosine",
                    "quantization": "scalar"
                }]
            },
            name=index_name,
            type="vectorSearch"
        )
        try:
            coll.create_search_index(index)
            logging.info(f"Vector index '{index_name}' created/updated on '{collection_name}'.")
        except Exception as e:
            logging.error(f"Failed to create/update index: {e}")

    def drop_vector_index(self, collection_name: str, index_name: str = "vector_index"):
        coll = self.connect_collection(collection_name)
        try:
            coll.drop_search_index(index_name)
            logging.info(f"Vector index '{index_name}' dropped from '{collection_name}'.")
        except Exception as e:
            logging.error(f"Failed to drop index: {e}")

    def add_documents(self, collection_name: str, records: list[dict]):
        coll = self.connect_collection(collection_name)
        texts = [rec['text'] for rec in records]
        embeddings = list(self.embedder.embed(texts))
        docs = []
        for rec, emb in zip(records, embeddings):
            doc = {'text': rec['text'], 'embedding': emb.tolist()}
            if 'metadata' in rec:
                doc['metadata'] = rec['metadata']
            docs.append(doc)
        try:
            coll.insert_many(docs)
            logging.info(f"Inserted {len(docs)} docs into '{collection_name}'.")
        except Exception as e:
            logging.error(f"Error inserting docs: {e}")

    def semantic_search(self, collection_name: str, query: str, k: int = 10, num_candidates: int = 100):
        coll = self.connect_collection(collection_name)
        if 'vector_index' not in coll.list_search_indexes():
            logging.warning("Vector index does not exist.")
        query_emb = list(self.embedder.embed([query]))[0].tolist()
        pipeline = [
            {'$vectorSearch': {
                'index': 'vector_index',
                'queryVector': query_emb,
                'path': 'embedding',
                'numCandidates': num_candidates,
                'limit': k
            }},
            {'$project': {'text': 1, 'metadata': 1, '_id': 0}}
        ]
        try:
            results = list(coll.aggregate(pipeline))
            logging.info(f"Found {len(results)} results for query '{query}'.")
            return results
        except Exception as e:
            logging.error(f"Search error: {e}")
            return []
