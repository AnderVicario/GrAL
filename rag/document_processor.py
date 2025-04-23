from llama_parse import LlamaParse
import asyncio

class DocumentProcessor:
    def __init__(self):
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            parse_embedding=False
        )

    async def process_pdf(self, file_path: str):
        documents = await self.parser.aload_data(file_path)
        return "\n".join(doc.text for doc in documents)