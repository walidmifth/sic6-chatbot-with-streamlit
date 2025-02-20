import faiss
import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4
from PyPDF2 import PdfReader

class Index:
    def __init__(self, pdf):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))) #untuk mendapatkan length vectornya
        self.vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        self.text = None
        reader = PdfReader(pdf)
        self.text = []
        for page in reader.pages:
            text = page.extract_text()
            self.text.append(text)
        self.text = " ".join(self.text)
        
    def chunk_text(self):
        splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(self.text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        
    def save_index(self):
        self.vector_store.save_local("faiss_index")
    