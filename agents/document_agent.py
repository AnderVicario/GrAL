import re
from together import Together
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "data")

_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
_db = _client[MONGODB_DB]

class DocumentAgent:
    def __init__(self):
        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

    def select_final_analysis(self, filename, first_page_content, user_company_list):
        prompt = """
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
            company = match.group(1).strip()
        else:
            company = "No match found"

        return company
    
    def list_financial_entities(self):
        collections = _db.list_collection_names()
        companies = []
        for name in collections:
            company = re.sub(r"_", " ", name).title()
            companies.append(company)
        return companies