from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from RAG_Chatbot.components.retriever.retriever import QdrantRetriever
from RAG_Chatbot.components.LLM.LLM import LLMManager

# 1. Initialize our components
retriever = QdrantRetriever(limit=3)
llm = LLMManager().get_model()

# 2. Create the prompt
template = """Answer the question based only on the following medical context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. THE LCEL CHAIN
# This structure is what LangSmith "reads" to create the visual trace.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Usage
# Every step (retrieval, prompt formatting, LLM call) is now traced in LangSmith!
response = rag_chain.invoke("What is diabetes?")