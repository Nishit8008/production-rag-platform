"""
Document Ingestion Module
Handles loading and processing of documents from URLs
"""
from typing import List
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentIngestion:
    """Handles document loading and chunking operations"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ):
        """
        Initialize the document ingestion pipeline
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_urls(self, urls: List[str]) -> List[Document]:
        """
        Load documents from a list of URLs
        
        Args:
            urls: List of valid URLs to load
            
        Returns:
            List of loaded documents
            
        Raises:
            ValueError: If URLs list is empty
            Exception: If loading fails
        """
        if not urls:
            raise ValueError("URLs list cannot be empty")
        
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline: load URLs and split into chunks
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of document chunks ready for embedding
        """
        documents = self.load_urls(urls)
        chunks = self.split_documents(documents)
        return chunks


def validate_urls(urls: List[str]) -> List[str]:
    """
    Filter and validate URLs
    
    Args:
        urls: List of URLs (may contain empty strings)
        
    Returns:
        List of valid, non-empty URLs
    """
    return [url.strip() for url in urls if url and url.strip()]