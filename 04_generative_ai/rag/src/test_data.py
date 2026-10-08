# Import required libraries
import unittest
import os
from data import DocumentProcessor

# Test for Document Processing class
class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):

        # Path relative to the src/ directory where the tests are executed
        self.valid_pdf_path   = "../data/Evaluation_of_the_Robustness_of_Machine_Learning.pdf"
        self.invalid_pdf_path = "../data/missing_document.pdf"

    def test_initialization_file_not_found(self):

        # Ensures the class raises an error if the PDF is missing.
        with self.assertRaises(FileNotFoundError):
            DocumentProcessor(self.invalid_pdf_path)

    def test_document_processing_pipeline(self):

        # Validates that a real PDF is loaded and split into text chunks correctly.        
        if not os.path.exists(self.valid_pdf_path):
            self.skipTest(f"Test PDF not found at {self.valid_pdf_path}. Skipping test.")
            
        processor = DocumentProcessor(
            file_path     = self.valid_pdf_path, 
            chunk_size    = 1000, 
            chunk_overlap = 200
        )
        chunks = processor.process()
        
        # Verify chunks were generated
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 0, "The processor should return at least one chunk.")
        
        # Verify the structure of a LangChain Document object
        first_chunk = chunks[0]
        self.assertTrue(hasattr(first_chunk, 'page_content'), "Chunk is missing page_content attribute.")
        self.assertTrue(hasattr(first_chunk, 'metadata'), "Chunk is missing metadata attribute.")
        self.assertIn('source', first_chunk.metadata, "Metadata should contain the source file path.")
        self.assertIn('page', first_chunk.metadata, "Metadata should contain the page number.")

if __name__ == '__main__':
    unittest.main()

 