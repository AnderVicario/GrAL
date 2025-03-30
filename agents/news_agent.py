import requests
from bs4 import BeautifulSoup
import urllib.parse
from gnews import GNews
import logging

logging.basicConfig(level=logging.INFO)

class NewsAnalysisAgent:
    def __init__(self, entity, search_terms, primary_language, date_range, 
                 sector=None, country=None, advanced_mode=False, max_results=20, search_mode="scraping"):
        self.entity = entity
        self.sector = sector
        self.country = country
        self.search_terms = search_terms
        self.primary_language = primary_language
        self.date_range = date_range
        self.advanced_mode = advanced_mode
        self.max_results = max_results
        # search_mode: "gnews" para usar la API o "scraping" para usar scraping personalizado
        self.search_mode = search_mode.lower()

    def process(self):
        articles = []
        queries = self._generate_queries()
        logging.info(f"Generated queries for entity '{self.entity}': {queries} Time range: {self.date_range}")
        count_for_entity = 0

        for query in queries:
            if count_for_entity >= self.max_results:
                logging.info(f"Reached limit for entity: {self.entity}. No further searches.")
                break

            new_articles = []

            # Si primary_language es 'es' y estamos en modo avanzado, se busca en español mediante scraping.
            if self.advanced_mode and self.primary_language.lower() == "es":
                spanish_articles = self._scrape_news(query, "es")
                logging.info(f"Found {len(spanish_articles)} articles in Spanish for query: {query}")
                new_articles += spanish_articles

            # Búsqueda en inglés siempre
            new_articles += self._search_english(query)
            logging.info(f"Found {len(new_articles)} articles in English for query: {query}")

            # Agregar solo los artículos necesarios hasta alcanzar el límite
            for art in new_articles[: self.max_results - count_for_entity]:
                articles.append(art)
                count_for_entity += 1

        return self._remove_duplicates(articles)

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
            if self.date_range:
                base_url += f"+when:{self.date_range}"
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
