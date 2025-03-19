class FinancialEntity:
    def __init__(self, name, ticker, entity_type, sector=None, country=None, primary_language=None, search_terms=None):
        self.name = name
        self.ticker = ticker
        self.entity_type = entity_type
        self.sector = sector
        self.country = country
        self.primary_language = primary_language
        self.search_terms = search_terms

    def __str__(self):
        return f"{self.entity_type.title()}: {self.name} ({self.ticker})"