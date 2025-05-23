import requests
from bs4 import BeautifulSoup
import country_converter as coco
import os
import json
from together import Together
import re


class MacroeconomicAnalysisAgent:
    EU_COUNTRIES = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT",
                    "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"}

    INFLUENCED_BY_USA = {"MX", "CA", "SV", "GT", "HN", "PA", "DO", "CO", "PE", "CL", "PH", "KR", "TW", "IL", "JP", "VN",
                         "TH", "CR", "EC"}

    INFLUENCED_BY_CHINA = {"PK", "LK", "LA", "KH", "MM", "ZW", "AO", "ZM", "KE", "ET", "VE", "AR", "BR", "ZA", "MY",
                           "ID"}

    INFLUENCED_BY_GERMANY = {"AT", "CZ", "PL", "SK", "HU", "NL", "BE", "SI", "LU"}

    INFLUENCED_BY_RUSSIA = {"BY", "KZ", "AM", "KG", "TJ", "MD", "RS", "BA", "GE"}

    INFLUENCED_BY_INDIA = {"NP", "BT", "LK", "MV", "BD", "MU", "FJ"}

    def __init__(self, name: str, ticker: str, sector: str, country: str, start_date: str = None, end_date: str = None):
        self.name = name
        self.ticker = ticker
        self.sector = sector
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

    def identify_entities(self):
        prompt = f"""
        You are a financial analyst specialized in macroeconomic analysis.
        Your task is to evaluate the general economic context and its potential impact on a specific entity.
        
        Entity: {self.name}, {self.ticker}
        Sector: {self.sector}
        Primary region of operations: {self.country}
        
        Current macroeconomic indicators for the country:
        {self.fetch_country_data()}
        
        Task:
        Based solely on the macroeconomic data provided:
        - Assess whether the current macroeconomic environment is favorable, neutral, or adverse for the company.
        - Justify your assessment based on the indicators.
        - If possible, identify potential risks and opportunities that could arise in the short to medium term for the company.
    
        Be specific and technical, as a professional financial economist would. Use clear, structured language in your response.
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

    def process(self):
        return {
            "report_type": "MacroeconomicAnalysis",
            "sector": self.sector,
            "country": self.country,
            "economic_indicators": {
                "inflacion": "2%",
                "PIB": "3%",
                "tasas_interes": "1.5%"
            }
        }

    def fetch_country_link(self, country_name, overwrite=False):
        filepath = "../data/countries.json"

        # Usar archivo existente si no se fuerza la actualización
        if os.path.exists(filepath) and not overwrite:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if item['country'] == coco.convert(names=country_name, to='name_short', not_found=None):
                        return item['url']
            return

        # Si no existe o se fuerza la actualización
        url = "https://tradingeconomics.com/countries"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            div_main = soup.find("div", {"id": "ctl00_ContentPlaceHolder1_ctl01_tableCountries"})

            if not div_main:
                print("Div principal no encontrado.")
                return

            os.makedirs("data", exist_ok=True)
            country_list = []

            for inner_div in div_main.find_all("div", recursive=False):
                table = inner_div.find("ul")
                if not table:
                    continue

                rows = table.find_all("li")
                if rows[0].get_text(strip=True) == "G20":
                    continue

                for row in rows[1:]:
                    a = row.find("a")
                    url = a['href']
                    text = a.get_text().lower()

                    if text != 'euro area':
                        country = coco.convert(names=text, to='name_short', not_found=None)
                        if country:
                            if country == coco.convert(names=country_name, to='name_short', not_found=None):
                                return url
                            country_list.append({"country": country, "url": url})
                country_list.append({"country": "Euro Area", "url": "/euro-area/indicators"})

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(country_list, f, indent=2, ensure_ascii=False)
        else:
            print(f"Error al acceder a la página: {response.status_code} {response.reason}")

    def get_influence_groups(self) -> list[str]:
        """
        Determina a qué bloques económicos pertenece el país.
        """
        groups = [self.country]
        # Siempre incluimos el propio país
        name_std = coco.convert(names=self.country, to='ISO2')

        if name_std in self.EU_COUNTRIES:
            groups.append("Euro Area")
        if name_std in self.INFLUENCED_BY_USA:
            groups.append("USA")
        if name_std in self.INFLUENCED_BY_CHINA:
            groups.append("China")
        if name_std in self.INFLUENCED_BY_GERMANY:
            groups.append("Germany")
        if name_std in self.INFLUENCED_BY_RUSSIA:
            groups.append("Russia")
        if name_std in self.INFLUENCED_BY_INDIA:
            groups.append("India")

        return groups

    def fetch_country_data(self, overwrite: bool = False):
        """
        Descarga datos macroeconómicos de self.country y de los países que lo influyen.
        """
        # Obtener los grupos de influencia
        response = {}
        influence_groups = self.get_influence_groups()
        print(influence_groups)

        # Iterar por cada bloque (incluyendo 'self')
        for country in influence_groups:
            if country != "Euro Area":
                iso2 = coco.convert(names=country, to='ISO2', not_found=None)
                filepath = f"../data/overview_{iso2}.json"
            else:
                iso2 = "EURO"
                filepath = f"../data/overview_{iso2}.json"

            # Cargar desde archivo si existe y no forzamos descarga
            if os.path.exists(filepath) and not overwrite:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = self._download_country_overview(country)
                if data is None:
                    continue
                # Guardar en JSON local
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            response[iso2] = data
        return response

    def _download_country_overview(self, country_name: str) -> list:
        """
        Método interno para descargar y parsear la tabla de Trading Economics.
        """
        # Construir URL basándonos en country_name
        url = f"https://tradingeconomics.com{self.fetch_country_link(country_name)}#overview"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error accediendo a {country_name}: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", {"class": "table table-hover"})
        if not table:
            print(f"Tabla no encontrada para {country_name}.")
            return None

        data = []
        tbody = table.find("tbody")
        if not tbody:
            return None

        for row in tbody.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 7:
                continue
            data.append({
                "indicator": cols[0].get_text(strip=True),
                "last": cols[1].get_text(strip=True),
                "previous": cols[2].get_text(strip=True),
                "highest": cols[3].get_text(strip=True),
                "lowest": cols[4].get_text(strip=True),
                "unit": cols[5].get_text(strip=True),
                "update_date": cols[6].get_text(strip=True)
            })
        return data


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    agent = MacroeconomicAnalysisAgent("Apple", "AAPL", "technology", "USA")
    print(agent.identify_entities())
