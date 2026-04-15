from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate


# STEP 1 — LOAD YOUR CHUNKS


chunks = [
    "Artificial Intelligence is the simulation of human intelligence in machines.",
    "Machine Learning is a subset of AI that allows systems to learn from data.",
    "RAG stands for Retrieval-Augmented Generation.",
    "Vector databases store embeddings for semantic search.",
    "Embeddings convert text into numerical representations."
]


# STEP 2 — CREATE EMBEDDINGS & VECTOR STORE

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("\n Vector store created and documents stored.\n")


# STEP 3 — BUILD SEMANTIC RETRIEVER


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(" Retriever created (Top-3 results).\n")


test_query = "What is RAG?"

retrieved_docs = retriever.invoke(test_query)

print(" Retriever Test Query:", test_query)
print(" Retrieved Chunks:")
for doc in retrieved_docs:
    print("-", doc.page_content)

print("\n" + "-" * 50 + "\n")


# STEP 4 — BUILD RAG CHAIN

llm = ChatOllama(model="qwen2.5:1.5b")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

def rag_chain(question):
    docs = retriever.invoke(question)
    
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = prompt_template.format(
        context=context,
        question=question
    )
    
    response = llm.invoke(prompt)
    
    return docs, response.content

print(" RAG chain ready.\n")


# STEP 5 — TEST WITH MULTIPLE QUESTIONS

questions = [
    "What is artificial intelligence?",
    "Explain embeddings.",
    "What does RAG stand for?"
]

for q in questions:
    print(" QUESTION:", q)
    
    docs, answer = rag_chain(q)
    
    print("\n Retrieved Chunks:")
    for doc in docs:
        print("-", doc.page_content)
    
    print("\n Answer:")
    print(answer)
    
    print("\n" + "=" * 60 + "\n")

print(" All tests completed.")