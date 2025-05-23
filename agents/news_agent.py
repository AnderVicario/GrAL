import json
import logging
import re
import urllib.parse

import requests
from bs4 import BeautifulSoup
from gnews import GNews
from together import Together


class NewsAnalysisAgent:
    def __init__(self, entity, search_terms, primary_language, start_date, end_date,
                 sector=None, country=None, advanced_mode=False, max_results=20, search_mode="scraping"):
        self.entity = entity
        self.sector = sector
        self.country = country
        self.search_terms = search_terms
        self.primary_language = primary_language
        self.start_date = start_date
        self.end_date   = end_date
        self.advanced_mode = advanced_mode
        self.max_results = max_results
        # search_mode: "gnews" APIa erabiltzeko edo "scraping" scraping bidez bilatzeko
        self.search_mode = search_mode.lower()
        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

    def process_and_chunk(self, base_metadata: dict, max_chars: int = 1500) -> list:
        articles = []
        queries = self._generate_queries()
        logging.info(f"Generated queries for entity '{self.entity}': {queries} Time range: {self.start_date} to {self.end_date}")
        count_for_entity = 0

        for query in queries:
            if count_for_entity >= self.max_results:
                logging.info(f"Reached limit for entity: {self.entity}. No further searches.")
                break

            new_articles = []

            # Primary_language 'es' bada eta modu aurreratuan bagaude, espainieraz bilatzen da scraping bidez.
            if self.advanced_mode and self.primary_language.lower() == "es":
                spanish_articles = self._scrape_news(query, "es")
                logging.info(f"Found {len(spanish_articles)} articles in Spanish for query: {query}")
                new_articles += spanish_articles

            # Bilaketa inglesez beti
            new_articles += self._search_english(query)
            logging.info(f"Found {len(new_articles)} articles in English for query: {query}")

            # Behar diren artikuluak bakarrik gehitu, mugara iritsi arte
            for art in new_articles[: self.max_results - count_for_entity]:
                articles.append(art)
                count_for_entity += 1

        articles = self._remove_duplicates(articles)

        # HEADLINES CHUNKS - Optimized JSON chunking
        chunks, current_chunk, current_size = [], [], 2
        for article in articles:
            article_size = len(json.dumps(article, ensure_ascii=False))
            if current_size + article_size > max_chars and current_chunk:
                chunks.append(json.dumps(current_chunk, ensure_ascii=False, indent=2))
                current_chunk, current_size = [article], 2 + article_size
            else:
                current_chunk.append(article)
                current_size += article_size + (1 if len(current_chunk) > 1 else 0)

        if current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False, indent=2))

        headline_outputs = [
            {
                "text": chunk,
                "metadata": {
                    **base_metadata,
                    "analysis_type": "news_headlines",
                    "chunk_number": i + 1,
                    "total_chunks": len(chunks),
                    "source": "NewsAgent",
                }
            }
            for i, chunk in enumerate(chunks)
        ]

        # SENTIMENT ANALYSIS CHUNK
        sentiment_result = self._sentiment_analysis(articles)
        sentiment_output = {
            "text": sentiment_result,
            "metadata": {
                **base_metadata,
                "analysis_type": "news_sentiment_analysis",
                "chunk_number": 1,
                "total_chunks": 1,
                "source": "NewsAgent",
            }
        }

        return headline_outputs + [sentiment_output]

    def _generate_queries(self):
        queries = [self.entity]

        if self.country:
            if isinstance(self.country, list):
                for c in self.country:
                    queries.append(f"{self.entity} {c}")
            else:
                queries.append(f"{self.entity} {self.country}")

        if self.sector:
            if isinstance(self.sector, list):
                for s in self.sector:
                    queries.append(f"{self.entity} {s}")
            else:
                queries.append(f"{self.entity} {self.sector}")

        if self.advanced_mode:
            if isinstance(self.search_terms, list):
                for term in self.search_terms:
                    queries.append(f"{self.entity} {term}")
            elif isinstance(self.search_terms, str):
                terms = [term.strip() for term in self.search_terms.split(',')]
                for term in terms:
                    queries.append(f"{self.entity} {term}")
            else:
                queries.append(f"{self.entity} {self.search_terms}")

        return queries

    def _search_english(self, query):
        if self.search_mode == "gnews":
            return self._search_gnews(query, "en")
        else:
            return self._scrape_news(query, "en")

    def _search_gnews(self, query, language):
        try:
            client = GNews(
                language=language,
                country=self.country if not isinstance(self.country, list) else None,
                max_results=self.max_results,
                period=self.date_range
            )
            return client.get_news(query)
        except Exception as e:
            logging.error(f"Error searching '{query}' in {language} using GNews: {e}")
            return []

    def _scrape_news(self, query, language):
        try:
            query_encoded = urllib.parse.quote_plus(query)
            base_url = f"https://news.google.com/rss/search?q={query_encoded}"
            if self.start_date and self.end_date:
                base_url += f"+after:{self.start_date}+before:{self.end_date}"
            if language.lower() == "es":
                url = base_url + "&hl=es&gl=ES&ceid=ES:es"
            else:
                url = base_url + "&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url)
            articles = []
            if response.status_code == 200:
                xml_data = response.content
                soup = BeautifulSoup(xml_data, "lxml-xml")
                items = soup.find_all("item")
                for item in items:
                    title = item.find("title").text if item.find("title") else "No title"
                    pub_date = item.find("pubDate").text if item.find("pubDate") else "No pubDate"
                    link = item.find("link").text if item.find("link") else "No link"
                    source = item.find("source").text if item.find("source") else "No source"
                    article = {
                        "title": title,
                        "pubDate": pub_date,
                        "source": source
                        # ,"url": link
                    }
                    articles.append(article)
            else:
                logging.error(f"Error fetching URL: {url} Status code: {response.status_code}")
            return articles
        except Exception as e:
            logging.error(f"Error scraping news for query '{query}' in {language}: {e}")
            return []

    def _remove_duplicates(self, articles):
        seen = set()
        unique_articles = []
        for article in articles:
            if article.get('title') not in seen:
                seen.add(article.get('title'))
                unique_articles.append(article)
        return unique_articles

    def _sentiment_analysis(self, headlines):
        print(headlines)
        prompt = f"""
        Your task is to analyze the sentiment of a series of financial news headlines. You will receive multiple headlines and must perform a sentiment analysis for each one according to these strict rules:

        1. For each headline, determine if the sentiment is **positive**, **neutral**, or **negative**. Classify the sentiment **only if it affects the entity mentioned by the user**.
        2. If a headline mistakenly mentions another company or deviates from the financial topic for any reason, classify it as **neutral**.
        
        ## OUTPUT REQUIREMENTS: At the end, count and output ONLY the total number of positive, neutral, and negative headlines in the following format: "Positives: X, Neutrals: Y, Negatives: Z". Do NOT include any additional text, explanations, or extra new lines in your output. Ensure that the output is provided exactly once without any repetition.

        User Entity:
        {self.entity}
        User Headlines:
        {headlines}

        Sentiment Analysis:
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
        pos_match = re.search(r"Positives:\s*(\d+)", response_text)
        neu_match = re.search(r"Neutrals:\s*(\d+)", response_text)
        neg_match = re.search(r"Negatives:\s*(\d+)", response_text)

        if pos_match and neu_match and neg_match:
            pos_count = pos_match.group(1)
            neu_count = neu_match.group(1)
            neg_count = neg_match.group(1)
            return f"Positives: {pos_count}, Neutrals: {neu_count}, Negatives: {neg_count}"
        else:
            return response_text