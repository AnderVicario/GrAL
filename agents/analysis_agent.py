import re
import json
import csv
from datetime import datetime
from typing import List, Dict

from together import Together


class AnalysisAgent:
    """
    Finantza-analisi sakonak sortzeko agente adimenduna.
    Erabiltzailearen kontsultak eta testuinguruko informazioa erabiliz, 
    epe ertain-luzeko errentagarritasun aurreikuspenak egiten ditu.
    """

    def __init__(self, user_prompt: str, horizon: str, context: str = None):
        """
        AnalysisAgent-aren hasieratzailea.
        
        Args:
            user_prompt: Erabiltzailearen kontsulta
            horizon: Analisiaren denbora tartea
            context: Testuinguruko informazioa (aukerazkoa):
                    - Finantza-egoerak
                    - Sektore-txostenak
                    - AMIA (SWOT) analisiak
                    - KPI taulak
                    - etab.
        """
        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        self.user_prompt = user_prompt
        self.current_date = datetime.now()
        self.horizon = horizon
        self.context = context

    def generate_final_analysis(self) -> str:
        """
        Analisi osoa sortzen du LLM eredua erabiliz. 
        Emaitza hiru atal nagusitan egituratzen da:
        
        1. Laburpen Exekutiboa:
           - Helburuaren berrazalpena
           - Goi-mailako ondorioa
           
        2. Analisi Xehatua:
           - Adierazle Gakoak (5-7 metriken aukeraketa eta justifikazioa)
           - Emaitza Kuantitatiboak (NPV, IRR, estres-probak, ratio-konparaketak)
           - Ikuspegi Kualitatiboak (industria, erregulazio-ingurunea, lehiakortasuna)
           
        3. Ondorioak eta Ekintza Plana:
           - Errentagarritasun probabilitatea: Altua/Ertaina/Baxua
           - 3-5 gomendio zehatz errendimendua hobetzeko edo arriskuak murrizteko
        
        Returns:
            str: Analisi osoa testu formatuan, pentsamendu-kateak kenduta
                 (<think>...</think> etiketak ezabatuta)
        
        Raises:
            ValueError: Testuinguruko informazio nahikorik ez badago
        
        Notes:
            - Datuen analisian oinarritutako ikuspegiak lehenesten dira
            - Formatua argia eta zehatza da, puntu edo taula txikiak erabiliz
            - Hizkera profesionala baina ulerterraza erabiltzen da
        """
        prompt = f"""
        You are a top-tier financial analysis agent with deep market insight, multistep reasoning, and advanced quantitative skills. Your goal is to assess whether any given financial entity is likely to be profitable over the medium to long term, and to deliver a clear, data-driven recommendation.

        SYSTEM INSTRUCTIONS:
        1. INPUTS YOU WILL RECEIVE
        - A **user query** describing what they want to know.
        - Zero or more **context attachments** (financial statements, sector reports, SWOT analyses, KPI spreadsheets, etc.).
        - A clearly defined **temporal horizon** (e.g., short-term, medium-term, long-term). Always align your analysis with this horizon.

        2. WORKFLOW
        1. **Context Validation**  
            - If no meaningful context or attachments are provided, respond:  
                “I need more detailed financial data or sector-specific documents to perform a thorough analysis. Please attach relevant reports or datasets.”
            - Otherwise, proceed to steps 2–5.
        2. **Executive Summary**  
            - In 2–3 sentences, restate the user’s objective, explicitly acknowledge the temporal horizon provided, and give your high-level verdict.
        3. **Key Indicators Selection**  
            - Identify the 5–7 most relevant metrics (e.g., ROE, EBITDA margin, debt/equity, free cash flow, beta, DCF inputs).  
            - For each, include a one-line justification.
        4. **Quantitative Analysis**  
            - Perform all necessary calculations (NPV, IRR, stress tests, ratio comparisons vs. peers).  
            - Display formulas or key assumptions if they materially affect the outcome.  
            - Present results as bullet points or a small table.
        5. **Qualitative Factors & Strategic Reasoning**  
            - Integrate industry dynamics, regulatory environment, competitive positioning, and macroeconomic trends.  
            - Use `<think>…</think>` tags around any private chains of thought or complex model assumptions.
        6. **Conclusion & Action Plan**  
            - Provide a clear verdict: “High probability of profitability,” “Moderate risk,” or “High risk of underperformance.”  
            - List 3–5 concrete recommendations to enhance returns or mitigate risks (e.g., capital structure changes, cost optimization, market diversification).

        3. RESPONSE FORMAT
        - **First**, any `<think>…</think>` internal reasoning (only if needed).
        - **Then**, the visible answer in three labeled sections:
            1. **Executive Summary**  
            2. **Detailed Analysis**  
                - Key Indicators  
                - Quantitative Results  
                - Qualitative Insights  
            3. **Conclusion & Action Plan**
        - Prioritize **data-driven insights** over general commentary.
        - Do **NOT** include any additional text, explanations, or new lines.

        4. STYLE GUIDELINES
        - Professional, precise, and jargon-light.
        - Justify each step clearly.
        - Use bullet points or small tables for clarity.

        -----

        User prompt: {self.user_prompt}
        Context: {self.context}
        Time horizon: {self.horizon}
        
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
        # print(full_response)
        return re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

    def evaluate_analysis(self, generated_analysis: str) -> Dict:
        """
        Envía el análisis previo a otro LLM para evaluarlo según la rúbrica:
        1. Clarity and coherence (0-10)
        2. Depth of quantitative analysis (0-10)
        3. Qualitative rigor (0-10)
        4. Usefulness of recommendations (0-10)
        5. Alignment with time horizon (0-10)
        6. Evidence and justification (0-10)
        Luego pide un overall_score y un summary (strengths/weaknesses).
        Devuelve un dict con la forma:
        {
          "clarity_coherence": {"score": int, "comment": str},
          ...,
          "overall_score": float,
          "summary": {"strengths": str, "weaknesses": str}
        }
        """
        rubric = """
        You are an expert agent specialized in evaluating financial analyses. Your task is to assess a financial analysis based on the following rubric:
        
        1. Clarity and coherence (0-10): Is the analysis well-structured and easy to understand?
        2. Depth of quantitative analysis (0-10): Are the financial indicators and calculations well explained?
        3. Qualitative rigor (0-10): Does it appropriately consider strategic, sectorial, and macroeconomic factors?
        4. Usefulness of recommendations (0-10): Are the recommendations clear, concrete, and actionable?
        5. Alignment with time horizon (0-10): Does the analysis match the requested time horizon?
        6. Evidence and justification (0-10): Are the conclusions well supported by data and reasoning?
        
        For each criterion, provide a numeric score and a brief comment.  
        At the end, include:
        - "overall_score": average of the six scores  
        - "summary": an object with "strengths" and "weaknesses"
        
        Please respond WITH EXACTLY ONE JSON object, for example:
        {
          "clarity_coherence": {"score": 8, "comment": "..."},
          "depth_quantitative": {"score": 7, "comment": "..."},
          "qualitative_rigor": {"score": 9, "comment": "..."},
          "usefulness_recommendations": {"score": 8, "comment": "..."},
          "alignment_horizon": {"score": 10, "comment": "..."},
          "evidence_justification": {"score": 7, "comment": "..."},
          "overall_score": 8.2,
          "summary": {
            "strengths": "...",
            "weaknesses": "..."
          }
        }
        
        Analysis to evaluate:
        """
        prompt = rubric + generated_analysis
        messages = [{"role": "user", "content": prompt}]
        resp = self.llm_client.chat.completions.create(
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
        full = ""
        for token in resp:
            if hasattr(token, "choices"):
                delta = token.choices[0].delta.content
                full += delta
        # Limpiar fences y pensar tags si los hubiera
        cleaned = re.sub(r"<think>.*?</think>", "", full, flags=re.DOTALL).strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:].strip()  # Elimina ```json y espacios iniciales
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()  # Elimina ``` y espacios iniciales

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()  # Elimina ``` final y espacios

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Manejar respuesta inválida del modelo
            return {
                "clarity_coherence": {"score": 0, "comment": "Invalid JSON response"},
                "depth_quantitative": {"score": 0, "comment": "Invalid JSON response"},
                "qualitative_rigor": {"score": 0, "comment": "Invalid JSON response"},
                "usefulness_recommendations": {"score": 0, "comment": "Invalid JSON response"},
                "alignment_horizon": {"score": 0, "comment": "Invalid JSON response"},
                "evidence_justification": {"score": 0, "comment": "Invalid JSON response"},
                "overall_score": 0,
                "summary": {"strengths": "", "weaknesses": "LLM returned invalid JSON"}
            }


def save_evaluation_to_csv(evaluation: Dict, csv_path: str, first_write: bool = False):
    """
    Guarda una sola evaluación en el CSV, añadiéndola al archivo existente o creando uno nuevo.

    Args:
        evaluation: Diccionario con la evaluación
        csv_path: Ruta del archivo CSV
        first_write: Si es True, escribe el encabezado
    """
    # Aplanar la evaluación
    row = {}
    for key, val in evaluation.items():
        if isinstance(val, dict) and "score" in val:
            row[f"{key}_score"] = val["score"]
            row[f"{key}_comment"] = val["comment"]
        elif key == "overall_score":
            row[key] = val
        elif key == "summary":
            row["strengths"] = val.get("strengths", "")
            row["weaknesses"] = val.get("weaknesses", "")
        else:
            # Para campos adicionales como domain, question, answer
            row[key] = val

    # Escribir en el CSV
    with open(csv_path, 'a' if not first_write else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if first_write:
            writer.writeheader()
        writer.writerow(row)
