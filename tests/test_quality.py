import asyncio
import csv, os
from main import get_application
from agents.analysis_agent import AnalysisAgent, save_evaluation_to_csv


async def evaluate_prompts(prompts_csv: str, output_csv: str):
    # 1) Leemos los prompts
    with open(prompts_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prompts = list(reader)

    app = get_application()
    first_write = True

    for row in prompts:
        domain = row['domain']
        question = row['question']

        try:
            # 2) Obtener la respuesta de la aplicación
            response = app.process_query(question)
            if asyncio.iscoroutine(response):
                response = await response

            # 3) Normalizar a texto
            if isinstance(response, list):
                formatted = []
                for item in response:
                    if isinstance(item, dict):
                        formatted.append(item.get("content", str(item)))
                    else:
                        formatted.append(str(item))
                answer_text = "\n".join(formatted)
            else:
                answer_text = str(response)

            # 4) Evaluar con AnalysisAgent
            agent = AnalysisAgent(
                user_prompt=question,
                horizon="medium-term",
                context=answer_text
            )
            evaluation = agent.evaluate_analysis(answer_text)
            print(evaluation)

            # 5) Añadimos metadata para el CSV
            evaluation["domain"] = domain
            evaluation["question"] = question
            evaluation["answer"] = answer_text

        except Exception as e:
            # En caso de error, guardamos un fallback
            evaluation = {
                "domain": domain,
                "question": question,
                "answer": "",
                "clarity_coherence": {"score": 0, "comment": str(e)},
                "depth_quantitative": {"score": 0, "comment": str(e)},
                "qualitative_rigor": {"score": 0, "comment": str(e)},
                "usefulness_recommendations": {"score": 0, "comment": str(e)},
                "alignment_horizon": {"score": 0, "comment": str(e)},
                "evidence_justification": {"score": 0, "comment": str(e)},
                "overall_score": 0,
                "summary": {"strengths": "", "weaknesses": str(e)}
            }

        # 6) Guardar esta evaluación en el CSV
        save_evaluation_to_csv(evaluation, output_csv, first_write)
        first_write = False  # Después de la primera escritura, solo añadiremos filas


def main():
    prompts_path = 'prompts.csv'
    output_path = 'test_quality_results.csv'

    # Eliminar archivo existente si lo hay para empezar fresco
    if os.path.exists(output_path):
        os.remove(output_path)

    asyncio.run(evaluate_prompts(prompts_path, output_path))


if __name__ == '__main__':
    main()
