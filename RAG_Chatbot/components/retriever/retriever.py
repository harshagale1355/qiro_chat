from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_classic.chains import RetrievalQA
from RAG_Chatbot.components.vector_store.vector import VectorStore
import sys
from RAG_Chatbot.components.LLM.LLM import LLMs
from langchain_classic.prompts import PromptTemplate

class QAchain:
    def retriever(self, strict_mode=True):
        try:
            vector_store = VectorStore()
            vector_doc = vector_store.vector()
            
            LLMS_instance = LLMs()
            LLMs_model = LLMS_instance.llms_model()
            
            
            qa_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are the Qiro Verse AI Agent. 

                STRICT OPERATING RULES:
                1. ONLY use the provided context to answer.
                2. If the answer is not in the context, you MUST say: "This information is not available in the provided documents."
                3. DO NOT introduce yourself. DO NOT say "I am an AI."
                4. DO NOT offer general help. 
                5. Response must be under 3 sentences.

                CONTEXT:
                {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
                QUESTION: {question}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            
            qa_prompt = PromptTemplate(
                template=qa_prompt_template,
                input_variables=["context", "question"]
            )
            
            # Groq is fast, so we can afford slightly higher 'k' 
            # for better context, provided the model's window allows it.
            qa_chain = RetrievalQA.from_chain_type(
                llm=LLMs_model,
                chain_type="stuff",
                retriever=vector_doc.as_retriever(
                    search_kwargs={
                        "k": 5, 
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": qa_prompt,
                    "verbose": False
                }
            )
            
            logging.info("Groq-powered strict retriever configured")
            return qa_chain
            
        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)