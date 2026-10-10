# Description: 
#
# 'VectorStoreManager' does below things
# 
# 1. Initializes the Embedding Model: 
#     - Loads the HuggingFace 'all-MiniLM-L6-v2' model, which is responsible for mathematically translating 
#       the raw text chunks into numerical vectors that capture semantic meaning.
# 
# 2. Builds and Persists the Database: 
#     - Takes the document chunks processed in 'data.py', passes them through the embedding model, and stores 
#       the resulting vectors in a local Chroma database. 
#       It saves this database to data/chroma_db folder so the PDF doesn't have to be re-embedded every 
#       time you run the script.
#
# 3. Configures the Search Engine (Retriever): 
#     - Exposes a retriever object that the LLM pipeline will use to query the database and fetch the 
#       paragraphs most relevant to the user's prompt.
#
# 4. Applies MMR (Maximum Marginal Relevance): 
#     - Specifically configures the retriever to balance relevance and diversity. Instead of just grabbing 
#       the top 3 most similar chunks (which might all say the exact same thing), MMR fetches a wider pool 
#       of 10 chunks, and selects the 3 most distinct ones to give the LLM a broader context.

# Import required libraries
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreManager:
    # Handles embedding generation, vector storage and retrieval configuration

    def __init__(self, persist_dir: str = "../data/chroma_db"):

        self.persist_dir     = persist_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def build_database(self, chunks):
        # Creates and saves a Chroma vector database from text chunks

        vector_db = Chroma.from_documents(
            documents = chunks, 
            embedding = self.embedding_model,
            persist_directory = self.persist_dir
        )

        return vector_db

    def get_retriever(self, vector_db, search_kwargs=None):
        # Configures and returns an MMR (Maximal Marginal Relevance) retriever from the vector database. 

        if search_kwargs is None: 
            search_kwargs = {"k": 3, "fetch_k": 10}

        retriever = vector_db.as_retriever(
            search_type   = "mmr",
            search_kwargs = search_kwargs
        )

        return retriever