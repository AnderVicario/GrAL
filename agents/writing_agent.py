import logging
import re
import colorlog
from datetime import datetime
from dotenv import load_dotenv
from together import Together

class MarkdownAgent:
    def __init__(self, user_text):
        load_dotenv()

        self.llm_client = Together()
        self.model_name = "meta-llama/Llama-Vision-Free"
        self.user_text = user_text
        self.current_date = datetime.now()

    def generate_markdown(self):
        prompt = f"""
        You are a writing assistant that formats plain text into clean Markdown.

        Rules:
        - Use only `##` headers if the user clearly provides a title or section header. DO NOT use `#` headers.
        - Do not invent titles, summaries, conclusions, or reasoning, ONLY convert to Markdown.
        - Emphasize important keywords with **bold** and use *italics* for special terms.
        - Convert lists into Markdown lists with `-`, do not use `|`.
        - Preserve the original structure and relevant information; remove only null or invalid details.
        - Do not explain your formatting decisions; just output the formatted Markdown.

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