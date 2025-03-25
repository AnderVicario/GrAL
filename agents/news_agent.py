from gnews import GNews
import logging

logging.basicConfig(level=logging.INFO)

class NewsAnalysisAgent:
    def __init__(self, entity, search_terms, primary_language, date_range, 
                 sector=None, country=None, mode='basic', max_results=10):
        self.entity = entity
        self.sector = sector
        self.country = country
        self.search_terms = search_terms
        self.primary_language = primary_language
        self.date_range = date_range
        self.mode = mode.lower()
        self.max_results = max_results

    def process(self):
        articles = []
        queries = self._generate_queries()
        
        for query in queries:
            articles += self._search_english(query)
            
            if self.mode == 'advanced':
                articles += self._search_primary_language(query)
        
        return self._remove_duplicates(articles)

    def _generate_queries(self):
        queries = [self.entity]
        
        if self.country:
            queries.append(f"{self.entity} {self.country_name}")
        
        if self.sector:
            queries.append(f"{self.entity} {self.sector}")
        
        if self.mode == 'advanced':
            queries += [f"{self.entity} {term}" for term in self.search_terms]
        
        return queries

    def _search_primary_language(self, query):
        try:
            client = GNews(
                language=self.primary_language,
                country=self.country,
                max_results=self.max_results,
                period=self.date_range
            )
            return client.get_news(query)
        except Exception as e:
            logging.error(f"Error searching {query} in {self.primary_language}: {e}")
            return []

    def _search_english(self, query):
        try:
            client = GNews(
                language='en',
                country='US',
                max_results=self.max_results,
                period=self.date_range
            )
            return client.get_news(query)
        except Exception as e:
            logging.error(f"Error searching {query} in English: {e}")
            return []

    def _remove_duplicates(self, articles):
        seen = set()
        unique_articles = []
        for article in articles:
            if article['url'] not in seen:
                seen.add(article['url'])
                unique_articles.append(article)
        return unique_articles

    def update_parameters(self, max_results=None, mode=None):
        if max_results:
            self.max_results = max_results
        if mode:
            self.mode = mode.lower()