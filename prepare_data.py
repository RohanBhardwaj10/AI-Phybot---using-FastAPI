from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

path = Path(r"C:\Users\bhard\ROHAN [ML & DS]\GEN  AI\AI Phybot Prroject\data\physics2.pdf")
loader=PyPDFLoader(path)
docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=100,
    length_function=len
)
split_docs=text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

vectorstore = FAISS.from_documents(split_docs, embeddings)

vectorstore.save_local("faiss_index")

print(" Data preparation complete! FAISS index saved successfully.")