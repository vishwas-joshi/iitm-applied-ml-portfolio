import unittest
import tempfile
import gc
from langchain_core.documents import Document
from vector_store import VectorStoreManager

class TestVectorStoreManager(unittest.TestCase):

    def setUp(self):
        # Python 3.10+ allows ignoring cleanup errors to bypass Windows file locks
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.manager  = VectorStoreManager(persist_dir=self.temp_dir.name)

        # Create dummy documents
        self.dummy_chunks = [
            Document(page_content="Wind turbines generate renewable energy.", metadata={"page": 1}),
            Document(page_content="Machine learning surrogate models predict loads.", metadata={"page": 2}),
            Document(page_content="Aeroelastic simulations are computationally expensive.", metadata={"page": 3})
        ]
        self.vector_db = None  # Track DB reference for cleanup

    def test_build_database_and_retriever(self):
        # 1. Test database creation
        self.vector_db = self.manager.build_database(self.dummy_chunks)
        self.assertIsNotNone(self.vector_db, "The vector database should not be None.")
        
        # 2. Test basic similarity search to ensure embeddings work
        results = self.vector_db.similarity_search("wind turbine", k=1)
        self.assertEqual(len(results), 1, "Should return exactly one result.")
        self.assertIn("renewable energy", results[0].page_content, "Should match the most relevant dummy chunk.")
        
        # 3. Test retriever configuration
        retriever = self.manager.get_retriever(self.vector_db)
        self.assertIsNotNone(retriever, "The retriever should not be None.")
        self.assertEqual(retriever.search_type, "mmr", "Retriever search type should be set to MMR.")

    def tearDown(self):
        # Release the vector DB references cleanly
        if hasattr(self, 'vector_db') and self.vector_db is not None:
            if hasattr(self.vector_db, '_client') and self.vector_db._client is not None:
                try:
                    self.vector_db._client.stop()
                except AttributeError:
                    pass
            del self.vector_db
        
        # Force garbage collection
        gc.collect()
        
        # This will now suppress WinError 32 and let the test pass!
        self.temp_dir.cleanup()

if __name__ == '__main__':
    unittest.main()
