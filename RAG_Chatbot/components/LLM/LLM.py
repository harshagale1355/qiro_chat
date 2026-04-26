from langchain_groq import ChatGroq  
from dotenv import load_dotenv
import os
from RAG_Chatbot.constant.constant_pipeline.__init import LLM_MODEL, TEMPERATURE
from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
import sys

load_dotenv()

class LLMs:
    def llms_model(self):
        try:
            llms = ChatGroq(
                model=LLM_MODEL,
                api_key=os.getenv("groq_api_key"),
                temperature=TEMPERATURE,
            )
            logging.info(f"Groq model {LLM_MODEL} loaded successfully")
            return llms
        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)