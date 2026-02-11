"""
Vector Store Module
Handles FAISS vector database operations
"""
import os
from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(
        self, 
        index_path: str = "faiss_index",
        embedding_model: str = "gemini-embedding-001"
    ):
        """
        Initialize the vector store manager
        
        Args:
            index_path: Path to store/load FAISS index
            embedding_model: Name of the embedding model to use
        """
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self._vectorstore: Optional[FAISS] = None
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create a new FAISS vector store from documents
        
        Args:
            documents: List of document chunks to embed
            
        Returns:
            FAISS vector store instance
        """
        self._vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self._vectorstore
    
    def save_vectorstore(self) -> None:
        """
        Save the current vector store to disk
        
        Raises:
            ValueError: If no vector store exists
        """
        if self._vectorstore is None:
            raise ValueError("No vector store to save. Create one first.")
        
        self._vectorstore.save_local(self.index_path)
    
    def load_vectorstore(self) -> FAISS:
        """
        Load vector store from disk
        
        Returns:
            Loaded FAISS vector store
            
        Raises:
            FileNotFoundError: If index doesn't exist
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Vector store not found at {self.index_path}. "
                "Please process URLs first."
            )
        
        self._vectorstore = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self._vectorstore
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever from the vector store
        
        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 4})
            
        Returns:
            Vector store retriever
            
        Raises:
            ValueError: If no vector store is loaded
        """
        if self._vectorstore is None:
            self._vectorstore = self.load_vectorstore()
        
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def index_exists(self) -> bool:
        """
        Check if FAISS index exists on disk
        
        Returns:
            True if index exists, False otherwise
        """
        return os.path.exists(self.index_path)
    
    def delete_index(self) -> None:
        """Delete the FAISS index from disk"""
        if os.path.exists(self.index_path):
            import shutil
            shutil.rmtree(self.index_path)