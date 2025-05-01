import datetime
import logging
import json
import re
import colorlog
from agents.analysis_agent import AnalysisAgent
from agents.news_agent import NewsAnalysisAgent
from agents.macro_agent import MacroeconomicAnalysisAgent
from agents.fundamental_agent import FundamentalAnalysisAgent
from agents.technical_agent import TechnicalAnalysisAgent
from agents.writing_agent import MarkdownAgent
from agents.document_agent import VectorMongoDB
from entities.financial_entity import FinancialEntity
from dotenv import load_dotenv
from datetime import datetime
from together import Together


class SearchAgent:
    def __init__(self, user_prompt):
        load_dotenv()
        self._configure_logging()

        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        self.user_prompt = user_prompt
        self.entities = []
        self.current_date = datetime.now()
        self.expiration_date = None
        self.date_range = None

    def _configure_logging(self):
        logger = colorlog.getLogger()
        if not logger.handlers:
            handler = colorlog.StreamHandler()
            handler.setFormatter(colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)-8s %(message)s%(reset)s",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            ))
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

    def _identify_entities(self):
        prompt = f"""
        You are a highly specialized financial assistant. Your task is to extract or suggest financial entities from the user's query. 
        The query can refer to one or several companies, and it may also include other types of financial entities such as cryptocurrencies, funds, ETFs, or other investment vehicles.
        If the query does not explicitly mention any entity, you must suggest several relevant entities based on the context of the question.

        For each financial entity, return a JSON object with the following keys:
        - "name": the full name of the entity
        - "ticker": the ticker or symbol (if available prefer the most popular or representative ticker for each entity, prioritizing the primary exchange of its home country for companies, and the standard symbol for cryptocurrencies, else null)
        - "entity_type": type of the entity (e.g., company, cryptocurrency, fund, ETF, etc.)
        - "sector": if applicable (can be null)
        - "country": if applicable (can be null)
        - "primary_language": the primary language for news or information (can be null). Use the ISO 639-1 language code (e.g., "en" for English).
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
            "primary_language": "en",
            "search_terms": "stock, investment, technology, innovation"
        }},
        {{
            "name": "Iberdrola, S.A.",
            "ticker": "IBE.MC",
            "entity_type": "company",
            "sector": "Energy",
            "country": "Spain",
            "primary_language": "es",
            "search_terms": "stock, investment, energy, renewable energy"
        }},
        {{
            "name": "Bitcoin",
            "ticker": "BTC",
            "entity_type": "cryptocurrency",
            "sector": null,
            "country": null,
            "primary_language": "en",
            "search_terms": "invest, crypto, digital currency"
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

    def _set_expiration_date(self):
        prompt = f"""
        Your task is to determine the appropriate expiration date for the financial query based on its investment horizon. Carefully analyze the query and follow these strict rules:

        1. If the query mentions something related with **a very short-term perspective** (e.g., "24h", "last hour", "intraday", "scalping", "high-frequency trading", "immediate action"), set the expiration date to **1 to 3 days** after today's date, depending on the urgency implied.
        2. If the query mentions something related with **a short-term perspective** (e.g., "this week", "short term", "day trading", "quick profits"), set the expiration date to **5 to 10 days** after today's date, allowing for market fluctuations.
        3. If the query mentions something related with **a long-term perspective** (e.g., "long term", "buy and hold", "retirement", "invest for the future"), set the expiration date to exactly **365 days** after today's date.
        4. If the query does not specify an investment horizon, default to **30 days** after today's date.

        ### Output IMPORTANT Requirements:
        - Return **ONLY** the expiration date.
        - The format must be strictly `YYYY-MM-DD` (e.g., `2025-04-29`).
        - Do **NOT** include any additional text, explanations, or new lines.

        Assume today's date is: {self.current_date}.

        User Query:
        {self.user_prompt}

        Expiration Date:
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
        expiration_date_str = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

        try:
            exp_date = datetime.strptime(expiration_date_str, '%Y-%m-%d')
            exp_date = exp_date.replace(
                hour=self.current_date.hour,
                minute=self.current_date.minute,
                second=self.current_date.second,
                microsecond=self.current_date.microsecond
            )
            self.expiration_date = exp_date.strftime('%Y-%m-%d %H:%M:%S')
            print(f"Fecha de expiración: {self.expiration_date}")

            # Kalkulatu tartea
            delta_days = (exp_date - self.current_date).days
            if delta_days < 1:
                self.date_range = f"{(exp_date - self.current_date).seconds // 3600}h"
            elif delta_days < 60:
                self.date_range = f"{delta_days}d"
            else:
                self.date_range = None
            
        except ValueError as e:
            logging.error(f"Error parsing expiration date: {e}")
            self.expiration_date = expiration_date_str
            self.date_range = None

    def _distil_query(self, entity):
        prompt = f"""
        You are a financial synthesis agent. Your task is to take two inputs:

        1. A **user prompt** – a financial or economic question or request, possibly broad or referring to multiple assets.  
        2. A **specific asset** – this can be a cryptocurrency, company, bank, index, country, or any economic entity.

        Your goal is to:

        - **Extract and reinterpret** the core intent behind the user's original prompt, focusing **exclusively** on the specific asset provided.
        - **Ignore comparisons, general questions, or references to other entities** – even if present in the user's query.
        - Output a **concise, precise, and actionable financial query** that relates solely to the given asset.

        Be professional, accurate, and specific. Do not invent facts. If the user’s intent is vague or overly general, assume a request for **general financial analysis** of the asset (including recent trends, relevant metrics, and news).

        ### Output IMPORTANT Requirements:
        - Return **ONLY** the refined prompt.
        - Do **NOT** include any additional text, explanations, or new lines.

        ---

        **Examples:**

        - **User Prompt:** *“How are cryptocurrencies performing so far this year?”*  
        **Asset:** *Ethereum*  
        **Refined Prompt:** `Provide a general analysis of Ethereum's performance this year, including price trends, key metrics, and major news.`

        - **User Prompt:** *“Compare the performance of Bitcoin and Ethereum this quarter.”*  
        **Asset:** *Ethereum*  
        **Refined Prompt:** `Analyze Ethereum's performance this quarter in isolation, including price evolution, trading volume, and relevant developments.`

        - **User Prompt:** *“Which tech companies are managing risk better during inflation?”*  
        **Asset:** *Apple Inc.*  
        **Refined Prompt:** `Assess how Apple Inc. is managing risk during the current inflationary period, including financial and strategic measures.`

        ---

        User Prompt:  
        {self.user_prompt}

        Asset:  
        {entity.name}

        Refined Prompt:
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
        response_text = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        logging.info(f"Refined Prompt: {response_text}")
        return response_text

    def _process_entities(self):
        self._set_expiration_date()
        response_text = self._identify_entities()
        logging.info(f"Respuesta del modelo: {response_text}")
        
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
            logging.error("No se encontró una estructura JSON válida en la respuesta.")

    def process_all(self, advanced_mode=False):
        self._process_entities()
        all_reports = []
        
        # Helper para dividir texto en chunks
        def chunk_text(text, max_chars=1500):
            chunks = []
            current_chunk = []
            current_length = 0
            
            # Dividir por párrafos primero
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if current_length + len(para) + 2 <= max_chars:  # +2 por los saltos de línea
                    current_chunk.append(para)
                    current_length += len(para) + 2
                else:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [para]
                        current_length = len(para)
                    else:
                        # Si un párrafo individual excede el máximo
                        chunks.append(para)
                        current_chunk = []
                        current_length = 0
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            return chunks

        # Procesar cada entidad
        for entity in self.entities:
            entity.create_vector_index()
            base_metadata = {
                "entity": entity.name,
                "ticker": entity.ticker,
                "entity_type": entity.entity_type,
                "report_date": self.current_date.isoformat(),
                "expiration_date": self.expiration_date if self.expiration_date else None
            }

            # 1. Análisis de Noticias
            news_agent = NewsAnalysisAgent(
                entity=entity.name,
                sector=entity.sector,
                country=entity.country,
                search_terms=entity.search_terms,
                primary_language=entity.primary_language,
                date_range=self.date_range,
                advanced_mode=advanced_mode
            )
            news_result = news_agent.process()
            news_markdown = MarkdownAgent(user_text=news_result).generate_markdown()
            
            # Dividir y subir chunks
            news_chunks = chunk_text(news_markdown)
            for i, chunk in enumerate(news_chunks):
                doc = {
                    "text": chunk,
                    "metadata": {
                        **base_metadata,
                        "analysis_type": "news",
                        "chunk_number": i+1,
                        "total_chunks": len(news_chunks),
                        "source": "NewsAnalysisAgent"
                    }
                }
                entity.add_documents([doc])

            # 2. Análisis Fundamental
            fundamental_agent = FundamentalAnalysisAgent(
                company=entity.name,
                ticker=entity.ticker,
                sector=entity.sector,
                date_range=self.date_range,
            )
            fundamental_result = fundamental_agent.process()
            fundamental_markdown = MarkdownAgent(user_text=fundamental_result).generate_markdown()
            
            # Dividir y subir chunks
            fund_chunks = chunk_text(fundamental_markdown)
            for i, chunk in enumerate(fund_chunks):
                doc = {
                    "text": chunk,
                    "metadata": {
                        **base_metadata,
                        "analysis_type": "fundamental",
                        "chunk_number": i+1,
                        "total_chunks": len(fund_chunks),
                        "source": "FundamentalAnalysisAgent"
                    }
                }
                entity.add_documents([doc])

            # Búsqueda Semántica optimizada
            entity_results = self._handle_semantic_search(entity)
            entity_results = entity_results["entity_results"] + entity_results["global_results"]
            analysis_agent = AnalysisAgent(
                user_prompt=self._distil_query(entity),
                date_range=self.date_range,
                context=entity_results
            )
            final_report = analysis_agent.generate_final_analysis()
            final_markdown = MarkdownAgent(user_text=final_report).generate_markdown()

            all_reports.append({
                "entity_name": entity.name,
                "content": final_markdown,
                "ticker": entity.ticker
            })

        return all_reports
    
    def _handle_semantic_search(self, entity):
        # Búsqueda en la entidad específica
        search_results_entity = entity.semantic_search(
            query=self.user_prompt,
            k=5,
            num_candidates=50
        )

        if search_results_entity:
            result_str_entity = "\n".join(
                f"{i+1}. {res['text'][:150]}..." 
                for i, res in enumerate(search_results_entity)
            )
            logging.info(f"\n🔍 Resultados para {entity._collection.name}:\n{result_str_entity}")

        # Búsqueda global
        global_entity = VectorMongoDB("global_reports")
        search_results_global = global_entity.semantic_search(
            query=self.user_prompt,
            k=5,
            num_candidates=50
        )

        if search_results_global:
            result_str_global = "\n".join(
                f"{i+1}. {res['text'][:150]}..." 
                for i, res in enumerate(search_results_global)
            )
            logging.info(f"\n🌍 Resultados globales:\n{result_str_global}")

        # Limpieza
        entity.drop_vector_index()
        global_entity.drop_vector_index("global_reports")

        # Puedes devolver ambos o combinarlos:
        return {
            "entity_results": search_results_entity,
            "global_results": search_results_global
        }
