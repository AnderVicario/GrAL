import os
import re
import requests
import yfinance as yf
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
import hashlib
import logging

# --------------------------
# Configuración y Utilidades
# --------------------------

# Configurar logging para ver mensajes en consola
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Conectar a MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client.financial_db
documents_collection = db.documents
cache_collection = db.cache  # Opcional: colección para cache de respuestas

# --------------------------
# Funciones Clave del Sistema
# --------------------------

def get_embedding(text: str) -> list:
    """
    Envía una petición a Together AI para obtener el embedding de un texto.
    """
    response = requests.post(
        "https://api.together.xyz/v1/embeddings",
        headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
        json={
            "input": text,
            "model": "togethercomputer/m2-bert-80M-8k-retrieval"  # Ajusta según corresponda
        }
    )
    if response.status_code != 200:
        logger.error("Error en get_embedding: " + response.text)
        return []
    data = response.json()
    return data['data'][0]['embedding']

def get_financial_data(ticker: str) -> str:
    """
    Descarga datos financieros de Yahoo Finance para el ticker indicado y formatea la información.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Extraer y formatear la información relevante
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        summary = info.get("longBusinessSummary", "Sin resumen disponible.")
        data_text = f"Ticker: {ticker}\nSector: {sector}\nIndustria: {industry}\nResumen: {summary}"
        return data_text
    except Exception as e:
        logger.error(f"Error al obtener datos para {ticker}: {e}")
        return f"Datos no disponibles para {ticker}"

def extract_tickers(query: str) -> list:
    """
    Extrae posibles tickers financieros de la consulta del usuario usando expresiones regulares
    y mapeo de nombres comunes a tickers.
    """
    # Patrón para tickers: secuencias de 1 a 5 letras mayúsculas (opcionalmente con guión)
    pattern = r"\b[A-Z]{1,5}(?:-[A-Z]{1,2})?\b"
    tickers = re.findall(pattern, query)
    
    # Mapeo de nombres comunes a tickers (puedes ampliar este diccionario)
    name_to_ticker = {
        "apple": "AAPL",
        "tesla": "TSLA",
        "amazon": "AMZN",
        "google": "GOOG",
        "microsoft": "MSFT"
    }
    for name, ticker in name_to_ticker.items():
        if name in query.lower():
            tickers.append(ticker)
    
    # Eliminar duplicados y devolver la lista
    return list(set(tickers))

def fetch_and_store_if_needed(tickers: list):
    """
    Para cada ticker extraído, verifica si existe información actualizada (menor a 24 horas)
    en MongoDB. Si no existe o está desactualizada, descarga los datos, genera el embedding
    y los almacena.
    """
    MAX_TICKERS_PER_QUERY = 3  # Evitar descargas masivas
    tickers = tickers[:MAX_TICKERS_PER_QUERY]
    
    for ticker in tickers:
        # Buscar el documento más reciente para el ticker
        existing = documents_collection.find_one({"ticker": ticker}, sort=[("last_updated", -1)])
        
        if not existing or (datetime.now() - existing["last_updated"] > timedelta(hours=24)):
            logger.info(f"Descargando datos para {ticker}")
            data_text = get_financial_data(ticker)
            embedding = get_embedding(data_text)
            document = {
                "ticker": ticker,
                "text": data_text,
                "embedding": embedding,
                "last_updated": datetime.now()
            }
            documents_collection.insert_one(document)
            logger.info(f"Datos para {ticker} guardados/actualizados.")
        else:
            logger.info(f"Datos para {ticker} ya están actualizados.")

def search_similar_documents(query: str, top_k: int = 5) -> list:
    """
    Convierte la consulta en un vector y busca en MongoDB los documentos más similares
    usando una búsqueda vectorial.
    """
    query_embedding = get_embedding(query)
    pipeline = [{
        "$vectorSearch": {
            "index": "vector_index",  # Asegúrate de que este nombre coincide con el índice configurado en Atlas
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 100,
            "limit": top_k
        }
    }]
    results = documents_collection.aggregate(pipeline)
    return list(results)

def prepare_context(query: str) -> str:
    """
    Prepara un contexto concatenando los textos de los documentos similares encontrados en MongoDB.
    """
    results = search_similar_documents(query)
    context = "\n\n".join([doc["text"] for doc in results])
    return context

def get_cached_answer(query: str) -> str:
    """
    Verifica si ya existe una respuesta cacheada para la consulta (opcional).
    """
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cached = cache_collection.find_one({"hash": query_hash})
    if cached:
        return cached.get("answer")
    return None

def set_cached_answer(query: str, answer: str):
    """
    Almacena en cache la respuesta generada para una consulta.
    """
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_collection.insert_one({
        "hash": query_hash,
        "answer": answer,
        "timestamp": datetime.now()
    })

def llm_response(query: str, context: str) -> str:
    """
    Genera una respuesta utilizando Together AI LLM, pasando el contexto obtenido.
    """
    prompt = f"""
Eres un asistente financiero. Responde usando solo este contexto:
{context}

Pregunta: {query}
Respuesta:"""
    response = requests.post(
        "https://api.together.xyz/v1/completions",
        headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
        json={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Ajusta el modelo según tus necesidades
            "prompt": prompt,
            "max_tokens": 500
        }
    )
    if response.status_code != 200:
        logger.error("Error en llm_response: " + response.text)
        return "Lo siento, ocurrió un error al generar la respuesta."
    data = response.json()
    return data["choices"][0]["text"]

def generate_answer(query: str) -> str:
    """
    Función principal que integra el flujo:
      1. Extraer tickers de la consulta.
      2. Verificar y descargar datos de Yahoo Finance si es necesario.
      3. Realizar búsqueda vectorial para preparar el contexto.
      4. Generar la respuesta final usando el LLM.
      5. (Opcional) Cachear la respuesta.
    """
    # Verificar cache
    cached = get_cached_answer(query)
    if cached:
        logger.info("Respuesta cacheada encontrada.")
        return cached

    # Paso 1: Extraer tickers de la consulta
    tickers = extract_tickers(query)
    logger.info(f"Tickers extraídos: {tickers}")

    # Paso 2: Verificar y actualizar datos financieros si es necesario
    fetch_and_store_if_needed(tickers)

    # Paso 3: Preparar el contexto a partir de la búsqueda vectorial
    context = prepare_context(query)
    logger.info("Contexto preparado.")

    # Paso 4: Generar la respuesta utilizando el LLM
    answer = llm_response(query, context)

    # Cachear la respuesta
    set_cached_answer(query, answer)

    return answer

# --------------------------
# Ejecución Principal
# --------------------------

if __name__ == "__main__":
    user_query = input("Ingresa tu consulta financiera: ")
    answer = generate_answer(user_query)
    print("\nRespuesta:", answer)
