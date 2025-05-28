import logging
import os
import re

from dotenv import load_dotenv
from fastembed import TextEmbedding
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.server_api import ServerApi

# Oinarrizko konfigurazioa kargatu
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
_db = _client[MONGODB_DB]


class FinancialEntity:
    """
    Entitate finantzarioen kudeaketarako klasea, bektore-bilaketa semantikoarekin.
    MongoDB datu-basean oinarrituta, dokumentuen biltegiratze eta bilaketa bektoriala 
    ahalbidetzen du.
    """

    def __init__(self, name: str, ticker: str, entity_type: str, sector: str = None, 
                 country: str = None, primary_language: str = None, search_terms: list = None):
        """
        FinancialEntity klasearen hasieratzailea.
        
        Args:
            name: Entitatearen izena
            ticker: Burtsako sinboloa
            entity_type: Entitate mota (adib. 'stock', 'etf', 'crypto')
            sector: Sektorea (aukerazkoa)
            country: Herrialdea (aukerazkoa)
            primary_language: Hizkuntza nagusia (aukerazkoa)
            search_terms: Bilaketa-termino gehigarriak (aukerazkoa)
            
        Notes:
            - MongoDB bilduma bat sortzen du entitatearen izenean oinarrituta
            - TextEmbedding modeloa erabiltzen du testuak bektorizatzeko
        """
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

        self._collection = coll

    def __str__(self) -> str:
        """
        Entitatearen string adierazpena.
        
        Returns:
            str: Formatua: "Mota: Izena (Ticker)"
        """
        return f"{self.entity_type.title()}: {self.name} ({self.ticker})"

    def create_vector_index(self):
        """
        Bektore-bilaketa indizea sortzen du MongoDB bilduman.
        
        Indizearen ezaugarriak:
        - 768 dimentsio
        - Kosinu antzekotasuna
        - Eskalar kuantizazioa
        
        Raises:
            Exception: Indizea sortzean erroreren bat gertatzen bada
        """

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
            self._collection.create_search_index(index)
            logging.info(f"Vector index 'vector_index' created or updated in '{self._collection.name}'.")
        except Exception as e:
            logging.error(f"Failed to create/update vector index: {e}")

    def drop_vector_index(self):
        """
        Bektore-bilaketa indizea ezabatzen du MongoDB bildumatik.
        
        Raises:
            Exception: Indizea ezabatzean erroreren bat gertatzen bada
        """

        try:
            self._collection.drop_search_index("vector_index")
            logging.info(f"Vector index 'vector_index' dropped from collection '{self._collection.name}'.")
        except Exception as e:
            logging.error(f"Failed to drop vector index: {e}")

    def add_documents(self, records: list):
        """
        Dokumentuak prozesatu eta MongoDB bilduman gordetzen ditu bektore-adierazpenarekin.
        
        Args:
            records: Dokumentuen zerrenda. Dokumentu bakoitza dict bat da:
                    {
                        'text': str,  # Dokumentuaren testua
                        'metadata': dict  # Metadatuak (aukerazkoa)
                    }
                    
        Process:
            1. Testu bakoitzarentzat bektore-adierazpena sortzen du
            2. Metadatuak gehitzen ditu (baldin badaude)
            3. MongoDB bilduman gordetzen ditu
            
        Raises:
            Exception: Dokumentuak txertatzean erroreren bat gertatzen bada
        """

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

    def semantic_search(self, query: str, k: int = 10, num_candidates: int = 100) -> list:
        """
        Bilaketa semantikoa burutzen du gordetako dokumentuetan.
        
        Args:
            query: Bilaketa-kontsulta
            k: Itzuli beharreko emaitza kopurua (lehenetsia: 10)
            num_candidates: Aztertu beharreko hautagai kopurua (lehenetsia: 100)
            
        Returns:
            list: Aurkitutako dokumentuen zerrenda, bakoitza dict formatuan:
                 {
                     'text': str,  # Dokumentuaren testua
                     'metadata': dict  # Metadatuak (baldin badaude)
                 }
                 
        Notes:
            - Kontsulta bektorizatu eta kosinu-antzekotasuna erabiltzen du bilaketarako
            - Indizerik ez badago, ohartarazpen bat sortzen du
            
        Raises:
            Exception: Bilaketan erroreren bat gertatzen bada
        """

        if 'vector_index' not in self._collection.list_search_indexes():
            logging.warning("Vector index does not exist.")

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