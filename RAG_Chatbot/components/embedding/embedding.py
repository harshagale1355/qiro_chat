from langchain_community.embeddings import HuggingFaceEmbeddings
from RAG_Chatbot.constant.constant_pipeline.__init import MODEL_NAME,MODEL_KWARG
from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
import sys

class embedding:
    def embedding_data(self):
        try:
            embedded=HuggingFaceEmbeddings(
                model_name=MODEL_NAME,
                model_kwargs=MODEL_KWARG
            )
            logging.info("Embedding model loaded successfully")
            return embedded
        except Exception as e:
            raise RAG_Chatbot_Exception(e,sys)
        