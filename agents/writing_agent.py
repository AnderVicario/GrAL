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
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        self.user_text = user_text
        self.current_date = datetime.now()

    def generate_markdown(self):
        prompt = f"""
        You are an advanced writing assistant specialized in formatting text into Markdown. Your task is to take the user's plain text and apply Markdown formatting intelligently and cleanly.

        Rules:
        - Never use `#` headers.
        - Use `##` only if the user clearly provided a title or section header (for example: isolated line, possibly in all caps, or obviously a section separator).
        - Do not create or invent titles or summaries from the content.
        - Convert important keywords into **bold** if they seem emphasized.
        - Use *italics* for words that indicate emphasis or special terms.
        - Convert lists into proper Markdown lists using `-` if items are listed.
        - Maintain the user's original structure as much as possible.
        - Ensure readability and clean structure without over-formatting.

        Assume today's date is: {self.current_date}.

        Example:

        User Input:
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