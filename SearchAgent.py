from together import Together
from dotenv import load_dotenv
from datetime import datetime
#from YahooAPI import YahooAPI
import re

load_dotenv()
class SeachAgent:
    def __init__(self):
        self.client = Together()
        self.today_date = datetime.today().isoformat()

    def generate_response(self, context: str, question: str, LLM_MODEL: str) -> str:
        prompt = f"""
You are a highly specialized financial assistant. For every financial query, your response MUST consist ONLY of exactly three parts, in the exact order below, with no additional text or commentary:

1. **Company Identifier**: Provide the official stock ticker (in uppercase) associated with the question. If the user does not provide a company, it will generate automatic suggestions based on the context.
2. **Expiration Date**: Provide the expiration date for the information's relevance, relative to today's date, in the format YYYY/MM/DD. Use the following rules based on today's date, which is provided:
   - If the query indicates immediate or short-term information (e.g., contains words such as "today", "current", "immediate", "short term", "24 hours"), set the expiration date to tomorrow's date.
   - If the query indicates a long-term period (e.g., contains phrases like "5 years", "long term", "extended period"), set the expiration date to today's date plus the specified period (e.g., if "5 years" is mentioned, add 5 years).
   - If the query requests historical data (e.g., contains "historical", "past", or refers to a past date), set the expiration date to a date in the past (e.g., yesterday's date) to indicate that the information is no longer current.
3. **Structured RAG Queries**: Provide several concise, bullet-pointed queries that the RAG system can use to retrieve the necessary data. Each query should be on a new line.

Assume today's date is: {self.today_date}.

Example:
Question: Should I invest in Tesla today?
Answer:
1. TESLA
2. 2025/03/16  (Assuming today's date is 2025/03/15, and "today" implies immediate information valid until tomorrow)
3. - Current trend of Tesla stock over the last 24 hours
   - Recent news affecting Tesla’s stock price
   - Social media sentiment analysis for Tesla (related with the timestamp of the question)

Now, based on the following context and question, provide your answer following the above structure exactly, with no additional information.

Context:
{context}

Question:
{question}

Answer:
"""
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
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
        # Se procesa cada token recibido en streaming y se muestra en tiempo real.
        for token in response:
            if hasattr(token, 'choices'):
                content = token.choices[0].delta.content
                full_response += content
                #print(content, end='', flush=True)
        #print()  # Salto de línea al finalizar el streaming

        return full_response

# Ejemplo de uso:
if __name__ == "__main__":
    context = "Current financial markets, recent economic news, and investment analysis."
    question = "New companies with high growth potential in the technology sector for the next 5 years and small risk of investment."
    LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    engine = SeachAgent()
    full_response = engine.generate_response(context, question, LLM_MODEL)

    clean_text = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    print(clean_text)
    pattern = r"1\. (.+?)\s*2\. (.+?)\s*3\. (.+)"
    match = re.search(pattern, clean_text, re.DOTALL)

    if match:
        tickers = match.group(1).strip()  # Primera parte: tickers
        date = match.group(2).strip()  # Segunda parte: fecha
        analysis_points = match.group(3).strip()  # Tercera parte: análisis en listas

        # Convertir los valores en listas si es necesario
        tickers_list = [t.strip() for t in tickers.split(",")]
        analysis_list = [line.strip("- ").strip() for line in analysis_points.split("\n") if line.strip()]

        print("📌 Tickers:", tickers_list)
        print("📅 Fecha:", date)
        print("📊 Puntos de análisis:", analysis_list)
    else:
        print("No se encontró coincidencia.")

