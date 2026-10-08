# Import required libraries
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document Processing class
class DocumentProcessor:
    # Handles the ingestion and text splitting of PDF documents. 

    def __init__(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input document does not exist. Please check. - {file_path}")

        # Assign
        self.file_path     = file_path
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self):
        # Loads the document using PyPDFLoader

        #
        loader    = PyPDFLoader(self.file_path)
        documents = loader.load()

        return documents

    def split_text(self, documents):
        # Splits the loaded document pages into smaller, overlapping chunks. 

        #
        splitter        = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap=self.chunk_overlap)
        document_chunks = splitter.split_documents(documents)

        return document_chunks
    
    def process(self):
        # Orchaestrates loading & splitting and returns the final document chunks

        #
        documents = self.load_document()
        chunks    = self.split_text(documents)

        return chunks

    