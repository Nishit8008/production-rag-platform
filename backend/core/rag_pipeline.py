"""
RAG Pipeline Module
Handles the complete Retrieval-Augmented Generation pipeline
"""
from typing import List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from .vector_store import VectorStoreManager


class RAGPipeline:
    """Manages the complete RAG question-answering pipeline"""
    
    DEFAULT_PROMPT_TEMPLATE = """
Answer the question based ONLY on the provided context.

Context:
{context}

Question:
{question}
"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        prompt_template: str = None
    ):
        """
        Initialize the RAG pipeline
        
        Args:
            vector_store_manager: Instance of VectorStoreManager
            model_name: LLM model name
            temperature: Temperature for LLM generation
            prompt_template: Custom prompt template (optional)
        """
        self.vector_store_manager = vector_store_manager
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.prompt = ChatPromptTemplate.from_template(template)
        
        self._chain = None
    
    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """
        Format list of documents into a single string
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def build_chain(self):
        """
        Build the RAG chain using LangChain Expression Language (LCEL)
        
        Returns:
            Configured RAG chain
        """
        retriever = self.vector_store_manager.get_retriever()
        
        self._chain = (
            {
                "context": retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return self._chain
    
    def query(self, question: str) -> str:
        """
        Process a question and return an answer
        
        Args:
            question: User's question
            
        Returns:
            Generated answer based on retrieved context
        """
        if self._chain is None:
            self.build_chain()
        
        answer = self._chain.invoke(question)
        return answer
    
    def query_with_sources(
        self, 
        question: str,
        max_source_length: int = 500
    ) -> Tuple[str, List[dict]]:
        """
        Process a question and return answer with source documents
        
        Args:
            question: User's question
            max_source_length: Maximum length of source text to return
            
        Returns:
            Tuple of (answer, list of source document dicts)
        """
        # Get answer
        answer = self.query(question)
        
        # Get source documents
        retriever = self.vector_store_manager.get_retriever()
        source_docs = retriever.invoke(question)
        
        # Format sources
        sources = [
            {
                "content": doc.page_content[:max_source_length],
                "metadata": doc.metadata
            }
            for doc in source_docs
        ]
        
        return answer, sources


def create_rag_pipeline(
    index_path: str = "faiss_index",
    embedding_model: str = "gemini-embedding-001",
    llm_model: str = "gemini-2.0-flash",
    temperature: float = 0.7
) -> RAGPipeline:
    """
    Factory function to create a complete RAG pipeline
    
    Args:
        index_path: Path to FAISS index
        embedding_model: Embedding model name
        llm_model: LLM model name
        temperature: LLM temperature
        
    Returns:
        Configured RAG pipeline instance
    """
    vector_store = VectorStoreManager(
        index_path=index_path,
        embedding_model=embedding_model
    )
    
    rag_pipeline = RAGPipeline(
        vector_store_manager=vector_store,
        model_name=llm_model,
        temperature=temperature
    )
    
    return rag_pipeline