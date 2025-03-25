from flask import Flask, render_template, request, jsonify
from agents.search_agents import get_news
from agents.macro_agent import analyze_macro
from agents.micro_agent import analyze_micro
from agents.technical_agent import analyze_technical
from agents.decision_agent import generate_recommendation
import json

app = Flask(__name__)

# Cargar memoria RAG (simulación con JSON)
def load_rag():
    try:
        with open("data/rag_storage.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        
        # Verificar si hay datos frescos en RAG
        rag_data = load_rag()
        if query in rag_data:
            result = rag_data[query]  # Usamos datos almacenados
        else:
            # Ejecutar agentes si no hay datos recientes
            news = get_news(query)
            macro = analyze_macro(query)
            micro = analyze_micro(query)
            technical = analyze_technical(query)

            # Generar recomendación final
            result = generate_recommendation(news, macro, micro, technical)
            
            # Guardar en RAG
            rag_data[query] = result
            with open("data/rag_storage.json", "w") as f:
                json.dump(rag_data, f)

        return render_template("results.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
