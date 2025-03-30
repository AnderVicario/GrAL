import logging
import re
import colorlog
from datetime import datetime
from dotenv import load_dotenv
from together import Together

class MarkdownAgent:
    def __init__(self, user_text):
        load_dotenv()
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)-8s %(message)s%(reset)s",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        )
        logger = colorlog.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        self.user_text = user_text
        self.current_date = datetime.now()

    def generate_markdown(self):
        prompt = f"""
        You are an advanced writing assistant specialized in formatting text into Markdown. Your task is to take the user's plain text and apply Markdown formatting intelligently and creatively in a structured way.
        
        - Use `#` for titles, `##` for subtitles, and `###` for smaller headers.
        - Convert important keywords into **bold** if they seem emphasized.
        - Use *italics* for words that indicate emphasis or special terms.
        - Convert lists into proper Markdown lists using `-`.
        - IMPORTANT Ensure the best readability and proper structuring based on context.

        Assume today's date is: {self.current_date}.

        Example:
        User Input:
        This is an example title
        This is an important concept that should be in bold. 
        
        Items in the list:
        Item one
        Item two
        Item three
        
        Expected Markdown Output:
        # This is an example title
        This is an **important concept** that should be in bold.
        
        Items in the list:
        - Item one
        - Item two
        - Item three

        User Input:
        {self.user_text}
        
        Markdown Output:
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
        full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        final_response = re.sub(r"^```markdown\s*", "", full_response, flags=re.DOTALL)
        final_response = re.sub(r"\s*```$", "", final_response, flags=re.DOTALL)
        return final_response.strip()