# rag.py
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# ---- Robust paths (works locally and on Render) ----
BASE = Path(__file__).resolve().parent
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", BASE / "vectorstore"))

# Build and save the vectorstore (run manually when you rebuild)
def create_vectorstore():
    print("📥 Loading documents...")
    src = BASE / "data" / "source.txt"
    loader = TextLoader(str(src))
    docs = loader.load()

    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print("🧠 Creating embeddings...")
    embed = OpenAIEmbeddings()

    print("📦 Building FAISS vector store...")
    db = FAISS.from_documents(chunks, embed)

    print(f"💾 Saving vectorstore to: {VECTORSTORE_DIR}")
    db.save_local(str(VECTORSTORE_DIR))

    print("✅ Vectorstore created.")

# Load the vectorstore and answer the question
def get_answer(query: str):
    print(f"🤖 Received query: {query}")
    embed = OpenAIEmbeddings()
    db = FAISS.load_local(str(VECTORSTORE_DIR), embed, allow_dangerous_deserialization=True)

    print("🔍 Searching for relevant documents...")
    docs = db.similarity_search(query)

    print("🧠 Querying OpenAI LLM...")
    llm = OpenAI(temperature=0, max_tokens=1200)
    chain = load_qa_chain(llm, chain_type="stuff")

    result = chain.run(input_documents=docs, question=query)
    print("✅ Answer generated.")
    return result
