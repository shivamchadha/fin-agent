import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from utils.ollama import model

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local("vector_db.index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

rag_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def rag_answer(user_question: str) -> str:
    result = rag_chain.invoke(user_question)
    return result["result"]

def retrieve_context(query: str) -> str:
    """
    Retrieve additional context documents from the FAISS index.
    Returns a concatenated string of relevant document excerpts.
    """
    retrieved_docs = retriever.get_relevant_documents(query)
    # Concatenate the retrieved text from each document.
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context
