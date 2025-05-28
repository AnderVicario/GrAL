import logging
import os
import re
from dataclasses import dataclass
from typing import List

import spacy
from dotenv import load_dotenv
from fastembed import TextEmbedding
from llama_parse import LlamaParse
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.server_api import ServerApi
from spacy.cli import download
from together import Together


@dataclass
class ChunkMetadata:
    """Chunk bakoitzaren metadatuak"""
    start_idx: int
    end_idx: int
    entities: List[dict] = None
    key_phrases: List[str] = None


# Konfigurazioa
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# MongoDB bezeroa
load_dotenv()
_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
_db = _client[MONGODB_DB]


class DocumentAgent:
    """
    Dokumentuen klasifikazioa eta entitate finantzarioen identifikazioa egiten duen agentea.
    """

    def __init__(self):
        """
        DocumentAgent-aren hasieratzailea.
        LLM eredua konfiguratzen du Together API bidez.
        """
        self.llm_client = Together(api_key=TOGETHER_API_KEY)
        self.model_name = LLM_MODEL

    def select_financial_entity(self, filename: str, first_page_content: str) -> str:
        """
        Dokumentu baten lehen orrialdean oinarrituta, erlazionatutako entitate finantzarioa identifikatzen du.
        
        Args:
            filename: Dokumentuaren izena
            first_page_content: Lehen orrialdearen edukia
            
        Returns:
            str: Identifikatutako entitatearen izena edo "No match found"
        """
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
        # print(f"Full response: {full_response}")
        clean_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        print(f"Response: {clean_response}")
        match = re.search(r"Company:\s*(.+)", clean_response)
        if match:
            return match.group(1).strip()
        return None


class DocumentProcessor:
    def __init__(self, use_spacy: bool = True, language_model: str = "en_core_web_sm"):
        """
        DocumentProcessor-aren hasieratzailea.
        
        Args:
            use_spacy: spaCy erabiliko den ala ez
            language_model: Erabili beharreko spaCy hizkuntza eredua
        """
        self.use_spacy = use_spacy
        self._nlp = None
        self._language_model = language_model

    def _ensure_model_installed(self) -> None:
        """
        Ziurtatzen du spaCy eredua instalatuta dagoela.
        Ez badago, automatikoki instalatzen du.
        """
        try:
            spacy.load(self._language_model)
        except OSError:
            logging.info(f"Instalatzen {self._language_model} eredua...")
            try:
                download(self._language_model)
                logging.info(f"{self._language_model} eredua ondo instalatu da")
            except Exception as e:
                logging.error(f"Errorea {self._language_model} eredua instalatzean: {str(e)}")
                raise

    @property
    def nlp(self):
        """Lazy loading spaCy ereduarentzat, instalazioa automatikoki kudeatuz"""
        if self.use_spacy and self._nlp is None:
            self._ensure_model_installed()
            self._nlp = spacy.load(self._language_model)
        return self._nlp

    def chunk_text(self, text: str, chunk_size: int = 1024, overlap: int = 128) -> List[dict]:
        """
        Testua zatitzen du aukeratutako metodoaren arabera.
        
        Args:
            text: Zatitu beharreko testua
            chunk_size: Zati bakoitzaren gehienezko tamaina
            overlap: Zatien arteko gainjartzea
            
        Returns:
            List[dict]: Testu zatiak eta beraien metadatuak
        """
        if self.use_spacy:
            return self._spacy_chunk(text, chunk_size, overlap)
        else:
            return self._basic_chunk(text, chunk_size, overlap)

    def _basic_chunk(self, text: str, chunk_size: int, overlap: int) -> List[dict]:
        """Oinarrizko zatiketarako metodoa"""

        def get_semantic_chunks(text: str) -> List[tuple]:
            lines = text.split('\n')
            chunks = []
            current_chunk = []
            current_size = 0
            start_idx = 0

            for line in lines:
                line_size = len(line)

                if (self._is_structural_break(line) and
                        current_size + line_size > chunk_size and
                        current_chunk):
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append((chunk_text, start_idx, start_idx + len(chunk_text)))
                    current_chunk = []
                    current_size = 0
                    start_idx += len(chunk_text) + 1

                current_chunk.append(line)
                current_size += line_size + 1

            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append((chunk_text, start_idx, start_idx + len(chunk_text)))

            return chunks

        semantic_chunks = get_semantic_chunks(text)
        return [
            {
                "text": chunk[0],
                "metadata": ChunkMetadata(
                    start_idx=chunk[1],
                    end_idx=chunk[2]
                )
            }
            for chunk in semantic_chunks
        ]

    def _spacy_chunk(self, text: str, chunk_size: int, overlap: int) -> List[dict]:
        """spaCy bidezko zatiketa metodoa"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_size = 0
        start_idx = 0

        for sent in doc.sents:
            sent_size = len(str(sent))

            if current_size + sent_size > chunk_size and current_chunk:
                chunk_text = " ".join(str(s) for s in current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": ChunkMetadata(
                        start_idx=start_idx,
                        end_idx=start_idx + len(chunk_text),
                        entities=self._extract_entities(current_chunk),
                        key_phrases=self._extract_key_phrases(current_chunk)
                    )
                })
                current_chunk = []
                current_size = 0
                start_idx += len(chunk_text) + 1

            current_chunk.append(sent)
            current_size += sent_size + 1

        if current_chunk:
            chunk_text = " ".join(str(s) for s in current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": ChunkMetadata(
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    entities=self._extract_entities(current_chunk),
                    key_phrases=self._extract_key_phrases(current_chunk)
                )
            })

        return self._apply_overlap(chunks, overlap)

    @staticmethod
    def _is_structural_break(line: str) -> bool:
        """Lerro bat egitura-haustura den erabakitzen du"""
        if not line.strip():
            return True
        if re.match(r'^\d+\.', line):
            return True
        if line.strip().endswith('.') and len(line) > 50:
            return True
        return False

    def _extract_entities(self, sentences) -> List[dict]:
        """Entitate garrantzitsuak ateratzen ditu"""
        if not self.use_spacy:
            return []

        entities = []
        for sent in sentences:
            for ent in sent.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        return entities

    def _extract_key_phrases(self, sentences) -> List[str]:
        """Gako-esaldiak ateratzen ditu"""
        if not self.use_spacy:
            return []

        key_phrases = []
        for sent in sentences:
            for chunk in sent.noun_chunks:
                if chunk.root.dep_ in {'nsubj', 'dobj'}:
                    key_phrases.append(str(chunk))
        return key_phrases

    @staticmethod
    def _apply_overlap(chunks: List[dict], overlap_size: int) -> List[dict]:
        """Zatien arteko gainjartzea aplikatzen du"""
        if not chunks or overlap_size <= 0:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
                continue

            prev_chunk = chunks[i - 1]

            # Entitate komunak bilatu
            if chunk["metadata"].entities and prev_chunk["metadata"].entities:
                common_entities = set(e["text"] for e in prev_chunk["metadata"].entities) & \
                                  set(e["text"] for e in chunk["metadata"].entities)
                if common_entities:
                    context = f"Aurreko testuingurua: {', '.join(common_entities)}\n\n"
                    chunk["text"] = context + chunk["text"]

            result.append(chunk)

        return result

    @staticmethod
    async def parse_pdf(file_path: str) -> List[str]:
        """
        PDF fitxategi bat prozesatu eta testua orrialdeka ateratzen du.
        
        Args:
            file_path: PDF fitxategiaren kokapena
            
        Returns:
            List[str]: Orrialde bakoitzeko testuaren zerrenda
        """
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            parse_mode="parse_page_with_agent",
            result_type="text"
        )
        pages = await parser.aload_data(file_path)
        return [p.text for p in pages]

    @staticmethod
    def chunk_text_old(text: str, chunk_size: int = 1024, overlap: int = 128) -> List[str]:
        """
        Testu luze bat zati txikiagotan banatzen du, gainjartzea kontuan hartuta.
        
        Args:
            text: Zatitu beharreko testua
            chunk_size: Zati bakoitzaren tamaina
            overlap: Zatien arteko gainjartzea
            
        Returns:
            List[str]: Testu zatien zerrenda
        """
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks


class VectorMongoDB:
    """
    MongoDB-rekin bektore bilaketa inplementatzen duen klasea.
    Dokumentuen bektorizazioa eta bilaketa semantikoa ahalbidetzen du.
    """

    def __init__(self, collection_name: str):
        """
        VectorMongoDB-ren hasieratzailea.
        
        Args:
            collection_name: MongoDB bildumaren izena
        """
        self.coll = _db[collection_name]
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

    def create_vector_index(self, index_name: str):
        """
        Bektore bilaketa indizea sortzen du MongoDB bilduman.
        
        Args:
            index_name: Indizearen izena
        """
        index = SearchIndexModel(
            definition={"fields": [{"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine",
                                    "quantization": "scalar"}]},
            name=index_name,
            type="vectorSearch"
        )
        try:
            self.coll.create_search_index(index)
            logging.info(f"Index '{index_name}' ready on '{self.coll.name}'")
        except Exception as e:
            logging.error(f"Error creating index: {e}")

    def drop_vector_index(self, index_name: str):
        """
        Bektore bilaketa indizea ezabatzen du.
        
        Args:
            index_name: Ezabatu beharreko indizearen izena
        """
        try:
            self.coll.drop_search_index(index_name)
            logging.info(f"Dropped index '{index_name}' from '{self.coll.name}'")
        except Exception as e:
            logging.error(f"Error dropping index: {e}")

    def add_documents(self, chunks: List[dict]):
        """
        Dokumentu zatiak bektorizatu eta MongoDB-n gordetzen ditu.

        Args:
            chunks: Dokumentu zatien zerrenda, bakoitza bere metadatuekin
        """
        texts = [str(chunk["text"]) for chunk in chunks]
        texts = [text for text in texts if text]

        if not texts:
            logging.warning("No valid texts to embed")
            return

        try:
            embeddings = list(self.embedder.embed(texts))

            docs = []
            for chunk, emb in zip(chunks, embeddings):
                doc = {
                    "text": str(chunk["text"]),
                    "embedding": emb.tolist(),
                    "metadata": chunk.get("metadata", {})
                }
                docs.append(doc)

            if docs:
                self.coll.insert_many(docs)
                logging.info(f"Inserted {len(docs)} docs in '{self.coll.name}'")

        except Exception as e:
            logging.error(f"Insertion error: {str(e)}")
            raise

    def semantic_search(self, query: str, k: int = 10, num_candidates: int = 100) -> List[dict]:
        """
        Bilaketa semantikoa burutzen du gordetako dokumentuetan.
        
        Args:
            query: Bilaketa kontsulta
            k: Itzuli beharreko emaitza kopurua
            num_candidates: Aztertu beharreko hautagai kopurua
            
        Returns:
            List[dict]: Aurkitutako dokumentu zatiak, antzekotasunaren arabera ordenatuta
        """
        q_emb = list(self.embedder.embed([query]))[0].tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "global_reports",
                    "queryVector": q_emb,
                    "path": "embedding",
                    "limit": k,
                    "numCandidates": num_candidates
                }
            },
            {
                "$project": {
                    "text": 1,
                    'metadata': 1,
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
    """
    Galdera-erantzun sistema inplementatzen duen klasea LLM eredua erabiliz.
    """

    def __init__(self):
        """
        QAEngine-ren hasieratzailea.
        Together API bidezko LLM eredua konfiguratzen du.
        """
        self.client = Together(api_key=TOGETHER_API_KEY)

    def generate_response(self, context: str, question: str) -> str:
        """
        Emandako testuinguruan oinarrituta, galderari erantzuna sortzen du.
        
        Args:
            context: Galderaren testuingurua
            question: Erantzun beharreko galdera
            
        Returns:
            str: LLM ereduak sortutako erantzuna
        """
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

# def main():
#     # Configuración de prueba
#     TEST_COLLECTION = "global_reports"
#     TEST_INDEX_NAME = "global_reports"

#     # 1. Inicializar la conexión
#     vector_db = VectorMongoDB(TEST_COLLECTION)

#     try:
#         # 2. Crear índice vectorial
#         # print("🛠️ Creando índice vectorial...")
#         # vector_db.create_vector_index(TEST_INDEX_NAME)

#         # time.sleep(5)

#         # # 3. Insertar documentos de prueba con metadatos
#         # test_documents = [
#         #     {
#         #         "text": "El cambio climático está afectando gravemente a los ecosistemas árticos",
#         #         "metadata": {
#         #             "source": "reporte_medioambiental_2023",
#         #             "tema": "cambio climático",
#         #             "pagina": 45
#         #         }
#         #     },
#         #     {
#         #         "text": "Las energías renovables representaron el 40% de la producción energética en 2024",
#         #         "metadata": {
#         #             "source": "informe_energético_Q2",
#         #             "tema": "energías renovables",
#         #             "region": "UE"
#         #         }
#         #     },
#         #     {
#         #         "text": "Nuevos avances en baterías de estado sólido para vehículos eléctricos",
#         #         "metadata": {
#         #             "source": "tech_report_jan",
#         #             "tema": "tecnología automotriz",
#         #             "empresa": "QuantumBatteries"
#         #         }
#         #     }
#         # ]

#         # print("📄 Insertando documentos de prueba...")
#         # vector_db.add_documents(test_documents)

#         # 4. Realizar búsqueda semántica
#         query = "NVIDIA Announces Financial Results for First Quarter"
#     #     print(f"\n🔍 Realizando búsqueda semántica para: '{query}'")
#         results = vector_db.semantic_search(query, k=2)

#     #     # 5. Mostrar resultados con metadatos
#         print(results)

#     #     # 6. Validación básica
#     #     assert len(results) > 0, "Error: No se encontraron resultados"
#     #     print("\n✅ Prueba exitosa: Se encontraron resultados relevantes")

#     # except Exception as e:
#     #     print(f"\n❌ Error en la prueba: {str(e)}")
#     finally:
#     #     # 7. Limpieza
#         print("\n🧹 Realizando limpieza...")
#     #     # vector_db.drop_vector_index(TEST_INDEX_NAME)

# if __name__ == "__main__":
#     main()
