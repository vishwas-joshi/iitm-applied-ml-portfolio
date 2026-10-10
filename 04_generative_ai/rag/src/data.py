# Description: 
#
# 'data.py' does below things:
#
# 1. Validates the File Path: 
#       - Checks if the target PDF document actually exists on the system before attempting to process it, 
#         preventing pipeline crashes.
#
# 2. Ingests the PDF: 
#       - Uses LangChain's PyPDFLoader to extract the raw text and metadata (like page numbers) from the 
#         input document.
#
# 3. Chunks the Text: 
#       - Passes the extracted document through a 'RecursiveCharacterTextSplitter' to break massive blocks of 
#         text into smaller, mathematically sized pieces (1000 characters). 
#       - This is necessary because LLMs have restricted context windows.
#
# 4. Applies Semantic Overlap: 
#       - Forces a 200-character overlap between sequential chunks. 
#       - This ensures that a sentence or concept cut in half by the chunker retains its surrounding context, 
#         preventing information loss at the seams.
#
# 5. Orchestrates the Output: 
#       - Combines these steps into a single process() pipeline that returns a clean list of structured 
#         Document objects, perfectly formatted for the vector database to embed.

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

    