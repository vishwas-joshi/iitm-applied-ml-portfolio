# Description:
#
# 1. Initializes the LLM: 
#       - Sets up the ChatGroq connection, pointing it to the chosen model (like Qwen or Llama 3) 
#         and setting the temperature to 0. 
#       - This strict temperature setting ensures the model provides deterministic, factual answers 
#         rather than creative or hallucinated ones.
# 
# 2. Defines the Prompt Template: 
#       - Establishes the exact instructions the LLM must follow. 
#       - It explicitly commands the model to base its answer only on the provided context, and 
#         critically, instructs it to admit "I don't know" if the context lacks the answer, 
#         which acts as a hard guardrail against hallucination.
# 
# 3. Formats the Context: 
#       - Includes a helper function that takes the individual chunk objects retrieved from the 
#         vector database and cleanly joins them together into one continuous, readable text string 
#         that the LLM can easily parse.
# 
# 4. Constructs the LCEL Pipeline: 
#       - Uses LangChain Expression Language to snap all the modular pieces together into a single, 
#         executable chain. 
#       - It dictates the exact flow of data: 
#           * it takes the user's question, 
#           * fetches the context via the retriever, 
#           * pipes both into the prompt template, 
#           * sends the filled prompt to the Groq LLM, 
#           * and finally parses the output into a clean string using StrOutputParser.

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 
class GenerationPipeline:
    # Handles LLM initialization, prompt construction, and the LCEL generation chain.
    
    #
    def __init__(self, retriever, model_name: str = "llama3-8b-8192"):

        self.retriever = retriever

        # Using a standard Groq model; you can update this to your specific Qwen model string
        self.llm    = ChatGroq(model_name=model_name, temperature=0)
        self.prompt = self._create_prompt()

    #   
    def _create_prompt(self):
        # Constructs the QA prompt template.
        template = """Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise, professional, and grounded in the context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        qa_prompt = PromptTemplate.from_template(template)

        return qa_prompt

    #
    def format_docs(self, docs):
        # Formats retrieved document chunks into a single string.

        #
        formatted_text = "\n\n".join(doc.page_content for doc in docs)

        return formatted_text

    #
    def build_chain(self):
        # Builds and returns the LCEL retrieval and generation chain.

        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain