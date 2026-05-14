import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq  
from langchain_core.runnables import ConfigurableField

from RAG_Chatbot.constant.constant_pipeline.__init import LLM_MODEL, TEMPERATURE
from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception

load_dotenv()

class LLMManager:
    """Manages the initialization of the ChatGroq model for LCEL chains."""

    def __init__(self):
        self.model_name = LLM_MODEL
        self.temperature = TEMPERATURE
        self.api_key = os.getenv("GROQ_API_KEY") 

    def get_model(self):
        """Returns a ChatGroq instance compatible with LCEL."""
        try:
            if not self.api_key:
                logging.error("Groq API Key not found in environment variables.")
                raise ValueError("GROQ_API_KEY is missing.")

            # Initializing ChatGroq
            llm = ChatGroq(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
            )
            
            logging.info(f"Groq model {self.model_name} loaded successfully")

            return llm

        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)

