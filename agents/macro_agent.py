import requests
from bs4 import BeautifulSoup
import country_converter as coco
import os
import json


class MacroeconomicAnalysisAgent:
    def __init__(self, name: str, ticker: str, sector: str, country: str, start_date: str = None, end_date: str = None):
        self.name = name
        self.ticker = ticker
        self.sector = sector
        self.country = country
        self.start_date = start_date
        self.end_date = end_date

        EU_COUNTRIES = {
            "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
            "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
            "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
            "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
            "Spain", "Sweden"
        }

        INFLUENCED_BY_USA = {
            "Mexico", "Canada", "El Salvador", "Guatemala", "Honduras", "Panama",
            "Dominican Republic", "Colombia", "Peru", "Chile", "Philippines",
            "South Korea", "Taiwan", "Israel", "Japan", "Vietnam",
            "Thailand", "Costa Rica", "Ecuador"
        }

        INFLUENCED_BY_CHINA = {
            "Pakistan", "Sri Lanka", "Laos", "Cambodia", "Myanmar", "Zimbabwe",
            "Angola", "Zambia", "Kenya", "Ethiopia", "Venezuela", "Argentina",
            "Brazil", "South Africa", "Malaysia", "Indonesia"
        }

        INFLUENCED_BY_GERMANY = {
            "Austria", "Czech Republic", "Poland", "Slovakia", "Hungary",
            "Netherlands", "Belgium", "Slovenia", "Luxembourg"
        }

        INFLUENCED_BY_RUSSIA = {
            "Belarus", "Kazakhstan", "Armenia", "Kyrgyzstan", "Tajikistan",
            "Moldova", "Serbia", "Bosnia and Herzegovina", "Georgia"
        }

        INFLUENCED_BY_INDIA = {
            "Nepal", "Bhutan", "Sri Lanka", "Maldives", "Bangladesh",
            "Mauritius", "Fiji"
        }

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

    def fetch_country_link(self, overwrite=False):
        filepath = "../data/countries.json"

        # Usar archivo existente si no se fuerza la actualización
        if os.path.exists(filepath) and not overwrite:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if item['country'] == coco.convert(names=self.country, to='name_short', not_found=None):
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
                            if country == coco.convert(names=self.country, to='name_short', not_found=None):
                                return url
                            country_list.append({"country": country, "url": url})
                country_list.append({"country": "Euro Area", "url": "/euro-area/indicators"})

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(country_list, f, indent=2, ensure_ascii=False)
        else:
            print(f"Error al acceder a la página: {response.status_code} {response.reason}")

    def fetch_country_data(self, overwrite=False):
        filepath = f"../data/overview_{coco.convert(names=self.country, to='ISO2', not_found=None)}.json"

        # Usar archivo existente si no se fuerza la actualización
        if os.path.exists(filepath) and not overwrite:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

        base_url = "https://tradingeconomics.com"
        url = base_url + self.fetch_country_link() + "#overview"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", {"class": "table table-hover"})

            if not table:
                print("Tabla principal no encontrada.")
                return

            data = []
            # Extraemos filas (tr) del tbody
            tbody = table.find("tbody")
            if not tbody:
                print("Cuerpo de tabla no encontrado.")
                return None

            rows = tbody.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 7:
                    continue  # fila inesperada

                # Extraemos nombre sin espacios extras
                indicator = cols[0].get_text(strip=True)
                last = cols[1].get_text(strip=True)
                previous = cols[2].get_text(strip=True)
                highest = cols[3].get_text(strip=True)
                lowest = cols[4].get_text(strip=True)
                unit = cols[5].get_text(strip=True)
                date = cols[6].get_text(strip=True)

                item = {
                    "indicator": indicator,
                    "last": last,
                    "previous": previous,
                    "highest": highest,
                    "lowest": lowest,
                    "unit": unit,
                    "update_date": date
                }
                data.append(item)

            # Guardamos en JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            print(f"Error al acceder a la página: {response.status_code} {response.reason}")


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    agent = MacroeconomicAnalysisAgent("Apple", "AAPL", "technology", "USA")
    agent.fetch_country_data()
