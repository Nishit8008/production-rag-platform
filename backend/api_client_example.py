"""
Example client for testing the FastAPI endpoints
"""
import requests
import json


class RockyBotClient:
    """Client for interacting with RockyBot API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def process_urls(self, urls: list[str]) -> dict:
        """
        Process URLs and create vector store
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Processing result
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/process-urls",
            json={"urls": urls}
        )
        response.raise_for_status()
        return response.json()
    
    def process_urls_async(self, urls: list[str]) -> dict:
        """
        Process URLs asynchronously
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Task submission result
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/process-urls-async",
            json={"urls": urls}
        )
        response.raise_for_status()
        return response.json()
    
    def query(self, question: str, include_sources: bool = True) -> dict:
        """
        Query the knowledge base
        
        Args:
            question: Question to ask
            include_sources: Whether to include source documents
            
        Returns:
            Answer and optional sources
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/query",
            json={
                "question": question,
                "include_sources": include_sources
            }
        )
        response.raise_for_status()
        return response.json()
    
    def clear_index(self) -> dict:
        """Clear the vector store index"""
        response = self.session.delete(f"{self.base_url}/api/v1/clear-index")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage"""
    client = RockyBotClient()
    
    # Check health
    print("Checking API health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Vectorstore ready: {health['vectorstore_ready']}\n")
    
    # Process URLs (example URLs - replace with actual news articles)
    urls = [
        "https://www.example.com/news/article1",
        "https://www.example.com/news/article2",
    ]
    
    print("Processing URLs...")
    try:
        result = client.process_urls(urls)
        print(f"Status: {result['status']}")
        print(f"Documents processed: {result['documents_processed']}\n")
    except requests.exceptions.HTTPError as e:
        print(f"Error processing URLs: {e}\n")
    
    # Query the knowledge base
    question = "What are the main topics discussed in the articles?"
    
    print(f"Asking: {question}")
    try:
        response = client.query(question, include_sources=True)
        print(f"\nAnswer:\n{response['answer']}\n")
        
        if response.get('sources'):
            print("Sources:")
            for i, source in enumerate(response['sources'], 1):
                print(f"\n{i}. {source['content'][:200]}...")
    except requests.exceptions.HTTPError as e:
        print(f"Error querying: {e}\n")
    
    # Clean up (optional)
    # print("\nClearing index...")
    # client.clear_index()


if __name__ == "__main__":
    main()