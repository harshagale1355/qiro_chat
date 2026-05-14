# test.py

from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

from RAG_Chatbot.components.retriever.retriever import QdrantRetriever
from RAG_Chatbot.components.LLM.LLM import LLMs


# -----------------------------------------------------------------------------
# Load Retriever
# -----------------------------------------------------------------------------

retriever = QdrantRetriever()


# -----------------------------------------------------------------------------
# Load LLM
# -----------------------------------------------------------------------------

llm = LLMs().llms_model()


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

qa_prompt_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are the InsightMed AI Agent.

STRICT RULES:
1. Answer ONLY from the provided context. If the context partially contains the answer, use the available information to answer briefly.
2. If answer is missing, say:
"This information is not available in the provided documents."
3. Keep answer under 3 sentences.

CONTEXT:
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

QUESTION:
{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

qa_prompt = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["context", "question"]
)


# -----------------------------------------------------------------------------
# QA Chain
# -----------------------------------------------------------------------------

qa_chain = RetrievalQA.from_chain_type(

    llm=llm,

    chain_type="stuff",

    retriever=retriever,

    return_source_documents=True,

    chain_type_kwargs={
        "prompt": qa_prompt
    }
)


# -----------------------------------------------------------------------------
# Query
# -----------------------------------------------------------------------------

query = "What is diabetes?"


response = qa_chain.invoke({
    "query": query
})


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print(f"QUERY: {query}")
print("=" * 80)

print("\nANSWER:\n")
print(response["result"])

print("\nSOURCES:\n")

for index, doc in enumerate(
    response["source_documents"],
    start=1
):

    print(f"\nResult {index}")

    print(f"Source: {doc.metadata.get('source')}")

    print(f"Topic: {doc.metadata.get('topic')}")

    print(f"Page: {doc.metadata.get('page')}")

    print(f"Score: {doc.metadata.get('score')}")

    print(f"\nText:\n{doc.page_content}")

    print("-" * 80)


# -----------------------------------------------------------------------------
# Close Client
# -----------------------------------------------------------------------------

retriever.close()