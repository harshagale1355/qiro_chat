import json
from RAG_Chatbot.components.LLM.LLM import LLMs

class BookingParser:
    def __init__(self):
        # Reuse your Groq instance
        self.llm = LLMs().llms_model()

    def extract_booking_info(self, message: str):
        prompt = f"""
        You are an extraction assistant. Extract booking details from this text: "{message}"
        
        Return ONLY a JSON object with these keys (use null if the information is missing):
        - "name": (string, extract names even if the message is just a single word)
        - "date": (string)
        - "time": (string)
        - "people": (integer)

        Rules:
        1. If information is missing, use null.
        2. If the text is just a solitary word or name (e.g. "Ramesh", "John"), assume it is for the "name" field.
        3. If the text is just a number (e.g. "4"), assume it is for the "people" field.
        4. Return ONLY the JSON object. No conversation. No markdown formatting.
    """
        response = self.llm.invoke(prompt)
        try:
            # Groq is fast, but sometimes adds markdown fences like ```json
            content = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(content)
        except Exception:
            return {"name": None, "date": None, "time": None, "people": None}