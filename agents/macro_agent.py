import json
import os
import re
from pathlib import Path

import country_converter as coco
import requests
from bs4 import BeautifulSoup
from together import Together


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

    BASE_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
    BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def __init__(self, name: str, ticker: str, sector: str, country: str, start_date: str = None, end_date: str = None):
        """
        Klasearen hasieratzailea.
        
        Args:
            name (str): Entitatearen izena
            ticker (str): Entitatearen burtsa kodea
            sector (str): Entitatearen sektorea
            country (str): Entitatearen jatorrizko herrialdea
            start_date (str, aukerakoa): Hasiera data
            end_date (str, aukerakoa): Amaiera data
        """
        self.name = name
        self.ticker = ticker
        self.sector = sector
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

    def _macro_analysis(self, data_to_analize):
        """
        Emandako datu makroekonomikoen analisia burutzen du LLM eredua erabiliz.
        
        Args:
            data_to_analize: Aztertu beharreko datu makroekonomikoak
            
        Returns:
            str: Analisiaren emaitza testuan
        """
        prompt = f"""
        You are a financial analyst specialized in macroeconomic analysis.
        Your task is to evaluate the general economic context and its potential impact on a specific entity.
        
        Entity: {self.name}, {self.ticker}
        Sector: {self.sector}
        Primary region of operations: {self.country}
        
        Current macroeconomic indicators for the country:
        {data_to_analize}
        
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

    def process_and_chunk(self, base_metadata: dict, max_chars: int = 1500) -> list:
        """
        Herrialdearen datu makroekonomikoak prozesatu eta zatitan banatzen ditu.
        
        Args:
            base_metadata (dict): Oinarrizko metadatuak
            max_chars (int): Zati bakoitzaren gehienezko karaktere kopurua
            
        Returns:
            list: Zatitutako datuen eta analisiaren lista
        """
        outputs = []

        if self.country is None:
            return outputs

        if not coco.convert(names=self.country, to='ISO2'):
            return outputs

        # 1. Datu makroekonomiko gordinak bilatu eta serializatu.
        raw_payload = self._fetch_country_data()
        raw_pretty = json.dumps(raw_payload, ensure_ascii=False, indent=4)

        # 2. Aztertu eta serializatu
        analysis_text = self._macro_analysis(raw_payload)

        def chunk_text(text: str) -> list:
            """Testua zatitu esaldi edo paragrafoen arabera."""
            sentences = re.split(r'(?<=[.!?])\s+', text)  # Divide por frases
            chunks = []
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 > max_chars and current:
                    chunks.append(current.strip())
                    current = sentence
                else:
                    current += " " + sentence if current else sentence
            if current:
                chunks.append(current.strip())
            return chunks

        def chunk_pretty(pretty_str):
            """JSON zatitu egituraren arabera."""
            lines = pretty_str.splitlines(keepends=True)
            chunks = []
            current = ""
            for line in lines:
                if len(current) + len(line) > max_chars and current:
                    chunks.append(current)
                    current = line
                else:
                    current += line
            if current:
                chunks.append(current)
            return chunks

        # Chunkak sortu payload bakoitzeko
        raw_chunks = chunk_pretty(raw_pretty)
        analysis_chunks = chunk_text(analysis_text)

        # Bildu chunk bakoitza metadatuekin eta itzuli.
        for i, chunk in enumerate(raw_chunks, 1):
            outputs.append({
                "text": chunk,
                "metadata": {
                    **base_metadata,
                    "analysis_type": "raw_country_data",
                    "chunk_number": i,
                    "total_chunks": len(raw_chunks),
                    "source": "MacroeconomicAnalysisAgent",
                }
            })
        for i, chunk in enumerate(analysis_chunks, 1):
            outputs.append({
                "text": chunk,
                "metadata": {
                    **base_metadata,
                    "analysis_type": "macroeconomic_assessment",
                    "chunk_number": i,
                    "total_chunks": len(analysis_chunks),
                    "source": "MacroeconomicAnalysisAgent",
                }
            })

        return outputs

    def _fetch_country_link(self, country_name, overwrite=False):
        """
        Trading Economics webgunetik herrialdearen URLa lortzen du.
        
        Args:
            country_name: Herrialdearen izena
            overwrite (bool): Fitxategian gainidatzi behar bada
            
        Returns:
            str: Herrialdearen URLa edo None
        """
        filepath = self.BASE_DATA_DIR / 'countries.json'

        # Lehendik dagoen fitxategia erabili, eguneratzea behartzen ez bada
        if filepath.exists() and not overwrite:
            with filepath.open('r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if item['country'] == coco.convert(names=country_name, to='name_short', not_found=None):
                        return item['url']
            return None

        # Ez badago edo eguneratzea behartzen bada:
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
                print("Main Div not found.")
                return None

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
                json.dump(country_list, f, indent=4, ensure_ascii=False)
                return None
        else:
            print(f"Error accessing the page: {response.status_code} {response.reason}")
            return None

    def _get_influence_groups(self) -> list[str]:
        """
        Herrialdea zein eragin-taldetan dagoen zehazten du.
        
        Returns:
            list[str]: Eragin-taldeen lista
        """
        groups = [self.country]
        # Beti sartzen dugu berezko herrialdea, noski
        name_std = coco.convert(names=self.country, to='ISO2')
        if not name_std:
            return groups

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

    def _fetch_country_data(self, overwrite: bool = False):
        """
        Herrialdearen datu makroekonomikoak jasotzen ditu.
        
        Args:
            overwrite (bool): Fitxategian gainidatzi behar bada
            
        Returns:
            dict: Herrialdearen datu makroekonomikoak
        """
        response = {}
        influence_groups = self._get_influence_groups()

        for country in influence_groups:
            iso2 = 'EURO' if country == 'Euro Area' else coco.convert(names=country, to='ISO2', not_found=None)
            filename = f"overview_{iso2}.json"
            filepath = self.BASE_DATA_DIR / filename

            # Kargatu edo deskargatu
            if filepath.exists() and not overwrite:
                with filepath.open('r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = self._download_country_overview(country)
                if data is None:
                    continue
                # Datuen direktorioa ziurtatu
                self.BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
                with filepath.open('w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            response[iso2] = data
        return response

    def _download_country_overview(self, country_name: str) -> list:
        """
        Trading Economics webgunetik herrialdearen ikuspegi orokorra deskargatzen du.
        
        Args:
            country_name (str): Herrialdearen izena
            
        Returns:
            list: Herrialdearen adierazle ekonomikoen lista edo None
        """
        # URLa country_name proiektuan oinarrituta eraiki
        url = f"https://tradingeconomics.com{self._fetch_country_link(country_name)}#overview"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Could not be found {country_name}: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", {"class": "table table-hover"})
        if not table:
            print(f"Could not be found table: {country_name}")
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


# if __name__ == '__main__':
#     from dotenv import load_dotenv
#     load_dotenv()
#
#     # agent = MacroeconomicAnalysisAgent("Apple", "AAPL", "technology", "USA")
#     agent = MacroeconomicAnalysisAgent("Iberdola", "IBE.MC", "energy", "Spain")
#     metadata = {
#                 'entity': 'Iberdola',
#                 'ticker': 'IBE.MC',
#                 'entity_type': 'stock',
#                 'expiration_date': None
#             }
#     print(agent.process_and_chunk(metadata))