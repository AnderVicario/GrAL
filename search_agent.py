import datetime
import logging
import json
from agents.news_agent import NewsAnalysisAgent
from agents.macro_agent import MacroeconomicAnalysisAgent
from agents.fundamental_agent import FundamentalAnalysisAgent
from agents.technical_agent import TechnicalAnalysisAgent
from entities.financial_entity import FinancialEntity
from dotenv import load_dotenv
from datetime import datetime, timedelta
from together import Together


class SearchAgent:
    def __init__(self, user_prompt):
        load_dotenv()
        logging.basicConfig(level=logging.INFO)

        self.llm_client = Together()
        self.model_name = "curie:ft-user-prompt"
        self.user_prompt = user_prompt
        self.entities = []
        self.current_date = datetime.now()
        self.expiration_date = None

    def identify_entities(self):
        # Prompt mejorado para capturar información de la consulta
        prompt = f"""
You are a highly specialized financial assistant. Your task is to extract or suggest financial entities from the user's query. 
The query can refer to one or several companies, and it may also include other types of financial entities such as cryptocurrencies, funds, ETFs, or other investment vehicles.
If the query does not explicitly mention any entity, you must suggest several relevant entities based on the context of the question.

For each financial entity, return a JSON object with the following keys:
- "name": the full name of the entity
- "ticker": the ticker or symbol (if available, else null)
- "entity_type": type of the entity (e.g., company, cryptocurrency, fund, ETF, etc.)
- "sector": if applicable (can be null)
- "country": if applicable (can be null)
- "primary_language": the primary language for news or information (can be null)
- "search_terms": additional search terms relevant to the entity (can be null)

Assume today's date is: {self.current_date}.

Example:
User Query:
"Tell me about Apple and Bitcoin trends."
Answer:
[
  {{
    "name": "Apple Inc.",
    "ticker": "AAPL",
    "entity_type": "company",
    "sector": "Technology",
    "country": "USA",
    "primary_language": "English",
    "search_terms": "stock, investment, technology, innovation"
  }},
  {{
    "name": "Bitcoin",
    "ticker": "BTC",
    "entity_type": "cryptocurrency",
    "sector": null,
    "country": null,
    "primary_language": "English",
    "search_terms": "invest?, crypto, digital currency"
  }}
]

User Query:
{self.user_prompt}

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
        return full_response

    def process_entities(self):
        """
        Este método procesa la respuesta del modelo para extraer la información de las entidades financieras,
        crea instancias de FinancialEntity y las añade a la lista self.entities. También asigna la fecha de expiración.
        """
        response_text = self.identify_entities()
        try:
            # Se asume que la respuesta está en formato JSON para facilitar el parseo
            entities_data = json.loads(response_text)
            for ent in entities_data:
                new_entity = FinancialEntity(
                    name=ent.get("name"),
                    ticker=ent.get("ticker"),
                    entity_type=ent.get("entity_type"),
                    sector=ent.get("sector"),
                    country=ent.get("country"),
                    primary_language=ent.get("primary_language"),
                    search_terms=ent.get("search_terms")
                )
                self.entities.append(new_entity)
        except json.JSONDecodeError:
            # En caso de que la respuesta no sea un JSON, se parsea línea a línea.
            for line in response_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    # Se asume el formato "EntityType: Name (Ticker)"
                    entity_type, rest = line.split(":", 1)
                    name_part, ticker_part = rest.strip().split("(", 1)
                    name = name_part.strip()
                    ticker = ticker_part.rstrip(")").strip()
                    new_entity = FinancialEntity(
                        name=name,
                        ticker=ticker,
                        entity_type=entity_type.strip()
                    )
                    self.entities.append(new_entity)
                except Exception as e:
                    logging.error(f"Error al parsear la línea '{line}': {e}")

        # Asignar la fecha de expiración, por ejemplo, 1 día después de la fecha actual.
        self.expiration_date = self.current_date + timedelta(days=1)
