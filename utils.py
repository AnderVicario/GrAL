from rag.vector_db import VectorMongoDB
from rag.document_processor import DocumentProcessor

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def process_file(file_path):
    processor = DocumentProcessor()
    parsed_text = await processor.process_pdf(file_path)
    vector_db = VectorMongoDB()
    vector_db.add_documents(
        vector_db._chunk_text(parsed_text)
    )
    # os.remove(file_path)  # Opcional: eliminar después de procesar