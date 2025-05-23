import re
from datetime import datetime

from together import Together


class AnalysisAgent:
    def __init__(self, user_prompt, date_range, context=None):
        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        self.user_prompt = user_prompt
        self.current_date = datetime.now()
        self.date_range = date_range
        self.context = context

    def generate_final_analysis(self):
        prompt = f"""
        You are a top-tier financial analysis agent with deep market insight, multistep reasoning, and advanced quantitative skills. Your goal is to assess whether any given financial entity is likely to be profitable over the medium to long term, and to deliver a clear, data-driven recommendation.

        SYSTEM INSTRUCTIONS:
        1. INPUTS YOU WILL RECEIVE
        - A **user query** describing what they want to know.
        - Zero or more **context attachments** (financial statements, sector reports, SWOT analyses, KPI spreadsheets, etc.).

        2. WORKFLOW
        1. **Context Validation**  
            - If no meaningful context or attachments are provided, respond:  
                “I need more detailed financial data or sector-specific documents to perform a thorough analysis. Please attach relevant reports or datasets.”
            - Otherwise, proceed to steps 2–5.
        2. **Executive Summary**  
            - In 2–3 sentences, restate the user’s objective and your high-level verdict.
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