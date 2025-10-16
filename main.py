import os 
from fastapi import FastAPI,UploadFile,Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn


##Langchain imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


##Load environement variables

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="📄 RAG Q&A Chatbot API")

##Load Pdfs and create reteriver

DATA_DIR="."
SELECTED_PDFS = ["Docker.pdf"]

def load_pdfs_from_backend():
    all_docs = []
    for pdf_path in SELECTED_PDFS:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

print("📚 Loading backend PDFs...")
all_docs = load_pdfs_from_backend()
print(f"✅ Loaded {len(all_docs)} pages from PDFs.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}
)

print("⚙️ Building FAISS vectorstore...")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()
print("✅ Vectorstore ready!")


## Setup LLM and Chains

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest user question, decide what to retrieve."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the context below to answer. "
               "If unsure, reply 'I don’t know based on the provided documents.'\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


##Chat Message History

chat_histories = {}

def get_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]


conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


##FASTAPI Endpoints

class ChatRequest(BaseModel):
    session_id: str
    question: str



@app.get("/")
def home():
    return {"message": "📄 RAG Q&A Chatbot API is running!"}


@app.post("/ask")
async def ask_question(request: ChatRequest):
    """Ask a question and get answer using RAG."""
    try:
        result = conversational_rag.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        return {"answer": result["answer"]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)