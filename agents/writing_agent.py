import re
from datetime import datetime

from dotenv import load_dotenv
from together import Together


class MarkdownAgent:
    """
    Testu arrunta Markdown formatuko testura bihurtzen duen agente adimenduna.
    Testuaren egitura mantenduz, Markdown formatuaren elementuak aplikatzen ditu 
    irakurgarritasuna hobetzeko.
    """

    def __init__(self, user_text: str, user_language: str = "english"):
        """
        MarkdownAgent-aren hasieratzailea.
        
        Args:
            user_text: Formateatu beharreko testu gordina
            
        Notes:
            LLM eredu bat erabiltzen du (meta-llama/Llama-Vision-Free) Together API bidez
        """
        load_dotenv()

        self.llm_client = Together()
        self.model_name = "meta-llama/Llama-Vision-Free"
        self.user_text = user_text
        self.user_language = user_language
        self.current_date = datetime.now()

    def generate_markdown(self) -> str:
        """
        Testu gordina Markdown formatura bihurtzen du arau zehatz batzuk jarraituz:
        
        Formateatze Arauak:
        1. Izenburuak:
           - Soilik '##' erabiltzen da erabiltzaileak argi ematen dituen izenburuetarako
           - Ez da izenburuen asmaketarik egiten
           
        2. Enfasi elementuak:
           - **letra lodia** hitz garrantzitsuetarako
           - *italikoa* termino berezietarako
           
        3. Zerrendak:
           - '-' ikurra erabiltzen da zerrenda elementuetarako
           - Ez da '|' ikurra erabiltzen
           
        4. Egitura:
           - Jatorrizko testuaren egitura mantentzen da
           - Balio gabeko edo baliogabeko xehetasunak soilik ezabatzen dira
           
        Returns:
            str: Markdown formatuan dagoen testua, garbia eta txukuna
                 - Markdown bloke etiketak (```) kenduta
                 - Hasierako eta amaierako hutsune gehigarriak kenduta
        
        Examples:
            Input:
            ```
            NIRE OHARRAK
            Hau oso ideia garrantzitsua da.
            
            Gustuko fruituak:
            sagarra
            banana
            gerezia
            ```
            
            Output:
            ```markdown
            ## NIRE OHARRAK
            Hau **oso ideia garrantzitsua** da.
            
            ### Gustuko fruituak:
            - sagarra
            - banana
            - gerezia
            ```
        """
        prompt = f"""
        You are a writing assistant that formats plain English text into clean Markdown.

        First, convert the text to Markdown following the rules below. Then, if the desired language is different from English (i.e., user language -> {self.user_language} != "en"), translate the final Markdown output to user language -> {self.user_language}.

        Rules:
        - Use only `##` headers if the user clearly provides a title or section header. DO NOT use `#` headers.
        - Do not invent titles, summaries, conclusions, or reasoning. ONLY convert to Markdown.
        - Emphasize important keywords with **bold** and use *italics* for special terms.
        - Convert lists into Markdown lists with `-`. Do not use tables or other formats.
        - Preserve the original structure and relevant information. Remove only null or irrelevant details.
        - Output only the final Markdown text, in the appropriate language. Do not explain or comment on your process.

        Assume today's date is: {self.current_date}.

        This is an example:
            Input:
            MY NOTES
            This is a very important idea.

            fruits I like:
            apple
            banana
            cherry

            Expected Markdown Output:
            ## MY NOTES
            This is a **very important** idea.

            ### fruits I like:
            - apple
            - banana
            - cherry

        User Input:
        {self.user_text}

        Markdown Output:
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        full_response = response.choices[0].message.content
        final_response = re.sub(r"^```markdown\s*", "", full_response, flags=re.DOTALL)
        final_response = re.sub(r"\s*```$", "", final_response, flags=re.DOTALL)
        return final_response.strip()