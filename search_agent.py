import datetime
import logging
import json
import re
import colorlog
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
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        )
        logger = colorlog.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
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
        return re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

    def process_entities(self):
        """
        Este método procesa la respuesta del modelo para extraer la información de las entidades financieras,
        crea instancias de FinancialEntity y las añade a la lista self.entities. También asigna la fecha de expiración.
        """
        response_text = self.identify_entities()
        logging.info(f"Respuesta del modelo: {response_text}")
        
        # Buscar y extraer solo la parte JSON de la respuesta
        json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            try:
                entities_data = json.loads(json_text)
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
                logging.info(f"Se procesaron {len(entities_data)} entidades correctamente.")
            except json.JSONDecodeError as e:
                logging.error(f"Error al decodificar JSON: {e}")
                self._fallback_parsing(response_text)
        else:
            logging.warning("No se encontró una estructura JSON válida en la respuesta.")
            self._fallback_parsing(response_text)

        # Asignar la fecha de expiración, por ejemplo, 1 día después de la fecha actual.
        self.expiration_date = self.current_date + timedelta(days=1)
    
    def _fallback_parsing(self, response_text):
        """
        Método de respaldo para extraer información cuando el formato JSON falla.
        Intenta extraer información utilizando expresiones regulares.
        """
        logging.info("Utilizando método de parseo alternativo.")
        
        # Buscar patrones como "Name": "Nombre", "Ticker": "XYZ"
        name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"', re.IGNORECASE)
        ticker_pattern = re.compile(r'"ticker"\s*:\s*"([^"]+)"', re.IGNORECASE)
        entity_type_pattern = re.compile(r'"entity_type"\s*:\s*"([^"]+)"', re.IGNORECASE)
        sector_pattern = re.compile(r'"sector"\s*:\s*"([^"]+)"', re.IGNORECASE)
        country_pattern = re.compile(r'"country"\s*:\s*"([^"]+)"', re.IGNORECASE)
        
        # Dividir por posibles separadores de entidades
        entity_blocks = re.split(r'},\s*{', response_text)
        
        for block in entity_blocks:
            name_match = name_pattern.search(block)
            ticker_match = ticker_pattern.search(block)
            
            if name_match or ticker_match:
                name = name_match.group(1) if name_match else None
                ticker = ticker_match.group(1) if ticker_match else None
                entity_type = entity_type_pattern.search(block).group(1) if entity_type_pattern.search(block) else "unknown"
                sector = sector_pattern.search(block).group(1) if sector_pattern.search(block) else None
                country = country_pattern.search(block).group(1) if country_pattern.search(block) else None
                
                new_entity = FinancialEntity(
                    name=name,
                    ticker=ticker,
                    entity_type=entity_type,
                    sector=sector,
                    country=country
                )
                self.entities.append(new_entity)
                logging.info(f"Entidad extraída por método alternativo: {name or ticker}")

    def process_all(self):
        """
        Este método procesa todas las entidades financieras identificadas y genera un reporte consolidado.
        """
        # Procesa las entidades utilizando el método anterior.
        self.process_entities()
        report_lines = []
        report_lines.append(f"Reporte generado el {self.current_date.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Fecha de expiración: {self.expiration_date.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("Entidades financieras identificadas:")
        if self.entities:
            for entity in self.entities:
                report_lines.append(str(entity))
        else:
            report_lines.append("No se identificaron entidades.")
        report = "\n".join(report_lines)
        return report