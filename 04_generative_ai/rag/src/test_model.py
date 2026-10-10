import unittest
import os
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from model import GenerationPipeline

def dummy_retriever_func(query):
    """Mocks the ChromaDB retriever output for testing purposes."""
    return [Document(page_content="Wind turbine loads are critical for design.", metadata={"page": 1})]

class TestGenerationPipeline(unittest.TestCase):
    def setUp(self):
        # Inject a fake API key so ChatGroq initializes without raising an error
        os.environ["GROQ_API_KEY"] = "gsk_dummy_key_for_testing_purposes_only"
        
        # Wrap the dummy function in a RunnableLambda so it supports the | operator
        self.dummy_retriever = RunnableLambda(dummy_retriever_func)
        self.pipeline = GenerationPipeline(retriever=self.dummy_retriever)

    def test_prompt_creation(self):
        """Validates that the prompt template expects the correct input variables."""
        prompt = self.pipeline._create_prompt()
        self.assertIn("context", prompt.input_variables, "Prompt must include {context}")
        self.assertIn("question", prompt.input_variables, "Prompt must include {question}")

    def test_document_formatting(self):
        """Validates that document chunks are formatted into a single string."""
        docs = dummy_retriever_func("test query")
        formatted_string = self.pipeline.format_docs(docs)
        self.assertEqual(formatted_string, "Wind turbine loads are critical for design.")

    def test_chain_building(self):
        """Validates that the LCEL chain compiles successfully."""
        chain = self.pipeline.build_chain()
        self.assertIsNotNone(chain, "The LCEL chain should not be None.")

if __name__ == '__main__':
    unittest.main()