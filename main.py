import asyncio
from typing import List, Dict

from agents.document_agent import DocumentAgent, DocumentProcessor, VectorMongoDB
from agents.search_agent import SearchAgent


class ApplicationLogic:
    def __init__(self):
        self.doc_agent = DocumentAgent()
        self.search_agent = None

    async def process_document(self, filepath: str, filename: str) -> None:
        """Dokumentua prozesatu eta datu-basean gorde"""
        try:
            pages = await DocumentProcessor.parse_pdf(filepath)
            first_page = pages[0] if pages else ""

            selected_company = self.doc_agent.select_financial_entity(
                filename,
                first_page
            )

            doc_processor = DocumentProcessor(use_spacy=True)
            full_text = "\n".join(pages)
            chunks = doc_processor.chunk_text(text=full_text)

            vector_db = VectorMongoDB("global_reports")

            for i, chunk in enumerate(chunks):
                doc = {
                    "text": chunk,
                    "metadata": {
                        "entity": selected_company,
                        "filename": filename,
                        "analysis_type": "document",
                        "chunk_number": i + 1,
                        "total_chunks": len(chunks),
                        "source": "DocumentAgent",
                    }
                }
                vector_db.add_documents([doc])

        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def process_query(self, prompt: str, advanced_mode: bool = False) -> List[Dict]:
        """Erabiltzailearen kontsulta prozesatu"""
        self.search_agent = SearchAgent(prompt)
        return self.search_agent.process_all(advanced_mode)


def get_application() -> ApplicationLogic:
    """Aplikazioaren logika instantzia bakarra sortu"""
    return ApplicationLogic()

