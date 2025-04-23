import os
import re
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO)

# Oinarrizko konfigurazioa kargatu
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
_db = _client[MONGODB_DB]

class FinancialEntity:
    def __init__(self, name, ticker, entity_type, sector=None, country=None, primary_language=None, search_terms=None):
        self.name = name
        self.ticker = ticker
        self.entity_type = entity_type
        self.sector = sector
        self.country = country
        self.primary_language = primary_language
        self.search_terms = search_terms

        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

        # Bildumaren izena normalizatu 
        raw = self.name.strip().lower()
        collection_name = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")

        # Bilduma eta indizea sortu edo lortu
        if collection_name not in _db.list_collection_names():
            coll = _db.create_collection(collection_name)
            logging.info(f"Collection '{collection_name}' created.")
        else:
            coll = _db[collection_name]
            logging.info(f"Collection '{collection_name}' already exists.")

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
            name="vector_index",
            type="vectorSearch"
        )
        try:
            coll.create_search_index(index)
            logging.info(f"Vector index 'vector_index' created or updated in '{collection_name}'.")
        except Exception as e:
            logging.error(f"Failed to create/update vector index: {e}")

        self._collection = coll

    def __str__(self):
        return f"{self.entity_type.title()}: {self.name} ({self.ticker})"

    def drop_vector_index(self):
        """Bektore-indizea bildumatik ezabatu"""
        
        try:
            self._collection.drop_search_index("vector_index")
            logging.info(f"Vector index 'vector_index' dropped from collection '{self._collection.name}'.")
        except Exception as e:
            logging.error(f"Failed to drop vector index: {e}")

    def add_documents(self, records):
        """Dokumentuak gehitu embedding-en bitartez."""

        texts = [rec['text'] for rec in records]
        embeddings = list(self.embedder.embed(texts))
        docs = []
        for rec, emb in zip(records, embeddings):
            doc = {
                'text': rec['text'],
                'embedding': emb.tolist()
            }
            if 'metadata' in rec:
                doc['metadata'] = rec['metadata']
            docs.append(doc)
        try:
            self._collection.insert_many(docs)
            logging.info(f"{len(docs)} documents inserted into '{self._collection.name}'.")
        except Exception as e:
            logging.error(f"Error inserting documents: {e}")

    def semantic_search(self, query, k=10, num_candidates=100):
        """Bilaketa semantikoa erabiltzailearen galderaren arabera."""

        query_embedding = list(self.embedder.embed([query]))[0].tolist()
        
        pipeline = [
            {
                '$vectorSearch': {
                    'index': 'vector_index',
                    'queryVector': query_embedding,
                    'path': 'embedding',
                    'numCandidates': num_candidates,
                    'limit': k
                }
            },
            {'$project': {'text': 1, 'metadata': 1, '_id': 0}}
        ]
        try:
            results = list(self._collection.aggregate(pipeline))
            logging.info(f"{len(results)} results found for query '{query}'.")
            return results
        except Exception as e:
            logging.error(f"Error during semantic search: {e}")
            return []


# if __name__ == "__main__":
#     # Example usage
#     entity = FinancialEntity(
#         name="Apple Inc.",
#         ticker="AAPL",
#         entity_type="company",
#         sector="Tech",
#         country="US"
#     )
    
#     # sample_records = [
#     #     {'text': 'Apple reports record quarterly revenue', 'metadata': {'source': 'news'}},
#     #     {'text': 'Apple unveils new iPhone model', 'metadata': {'source': 'press'}}
#     # ]
#     # entity.add_documents(sample_records)

#     query = 'new iPhone price'
#     results = entity.semantic_search(query, k=2)
#     print('Semantic results:')
#     for idx, doc in enumerate(results, 1):
#         print(f"{idx}. {doc['text']} | metadata: {doc.get('metadata')}")
    
#     # Optional: Cleanup index
#     entity.drop_vector_index()