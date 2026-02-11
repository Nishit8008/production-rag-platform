"""
FastAPI Microservice for RAG Application
Provides REST API endpoints for document ingestion and question answering
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

from core import DocumentIngestion, VectorStoreManager, RAGPipeline, validate_urls


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="RockyBot RAG API",
    description="News Research Tool API with RAG capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
FAISS_PATH = "faiss_index"

# Initialize components
ingestion = DocumentIngestion(chunk_size=1000, chunk_overlap=200)
vector_store = VectorStoreManager(index_path=FAISS_PATH)
rag_pipeline = RAGPipeline(vector_store_manager=vector_store)


# Request/Response Models
class URLProcessRequest(BaseModel):
    """Request model for URL processing"""
    urls: List[str] = Field(..., min_items=1, description="List of URLs to process")
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": [
                    "https://example.com/article1",
                    "https://example.com/article2"
                ]
            }
        }


class URLProcessResponse(BaseModel):
    """Response model for URL processing"""
    status: str
    message: str
    documents_processed: int


class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str = Field(..., min_length=1, description="Question to ask")
    include_sources: bool = Field(default=True, description="Include source documents")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main points of the article?",
                "include_sources": True
            }
        }


class SourceDocument(BaseModel):
    """Model for source document"""
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str
    sources: Optional[List[SourceDocument]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vectorstore_ready: bool
    message: str


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RockyBot RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process_urls": "/api/v1/process-urls",
            "query": "/api/v1/query",
            "clear_index": "/api/v1/clear-index"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    vectorstore_ready = vector_store.index_exists()
    
    return HealthResponse(
        status="healthy",
        vectorstore_ready=vectorstore_ready,
        message="Vector store is ready" if vectorstore_ready else "Vector store not initialized"
    )


@app.post("/api/v1/process-urls", response_model=URLProcessResponse)
async def process_urls(request: URLProcessRequest):
    """
    Process URLs and create vector store
    
    Args:
        request: URLProcessRequest with list of URLs
        
    Returns:
        URLProcessResponse with processing status
    """
    try:
        # Validate URLs
        valid_urls = validate_urls(request.urls)
        
        if not valid_urls:
            raise HTTPException(
                status_code=400,
                detail="No valid URLs provided"
            )
        
        logger.info(f"Processing {len(valid_urls)} URLs")
        
        # Process documents
        documents = ingestion.process_urls(valid_urls)
        
        # Create and save vector store
        vector_store.create_vectorstore(documents)
        vector_store.save_vectorstore()
        
        logger.info(f"Successfully processed {len(documents)} document chunks")
        
        return URLProcessResponse(
            status="success",
            message="URLs processed successfully",
            documents_processed=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error processing URLs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing URLs: {str(e)}"
        )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base
    
    Args:
        request: QueryRequest with question
        
    Returns:
        QueryResponse with answer and optional sources
    """
    try:
        # Check if vector store exists
        if not vector_store.index_exists():
            raise HTTPException(
                status_code=400,
                detail="Vector store not initialized. Please process URLs first."
            )
        
        logger.info(f"Processing query: {request.question}")
        
        # Process query
        if request.include_sources:
            answer, sources = rag_pipeline.query_with_sources(request.question)
            source_models = [
                SourceDocument(content=s["content"], metadata=s["metadata"])
                for s in sources
            ]
            return QueryResponse(answer=answer, sources=source_models)
        else:
            answer = rag_pipeline.query(request.question)
            return QueryResponse(answer=answer)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.delete("/api/v1/clear-index")
async def clear_index():
    """
    Clear the vector store index
    
    Returns:
        Status message
    """
    try:
        if vector_store.index_exists():
            vector_store.delete_index()
            logger.info("Vector store index cleared")
            return {"status": "success", "message": "Vector store cleared"}
        else:
            return {"status": "success", "message": "No vector store to clear"}
            
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing index: {str(e)}"
        )


# Optional: Background task for async processing
@app.post("/api/v1/process-urls-async")
async def process_urls_async(
    request: URLProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Process URLs asynchronously in the background
    
    Args:
        request: URLProcessRequest with list of URLs
        background_tasks: FastAPI background tasks
        
    Returns:
        Task submission confirmation
    """
    def process_task(urls: List[str]):
        try:
            valid_urls = validate_urls(urls)
            documents = ingestion.process_urls(valid_urls)
            vector_store.create_vectorstore(documents)
            vector_store.save_vectorstore()
            logger.info(f"Background task completed: {len(documents)} chunks processed")
        except Exception as e:
            logger.error(f"Background task failed: {str(e)}")
    
    valid_urls = validate_urls(request.urls)
    if not valid_urls:
        raise HTTPException(status_code=400, detail="No valid URLs provided")
    
    background_tasks.add_task(process_task, valid_urls)
    
    return {
        "status": "processing",
        "message": f"Processing {len(valid_urls)} URLs in background",
        "urls": valid_urls
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)