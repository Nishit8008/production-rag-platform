"""
Refactored Streamlit Application
Communicates with FastAPI backend for all RAG operations
"""
import streamlit as st
import requests
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RockyBot: News Research Tool",
    page_icon="📈",
    layout="wide"
)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Configuration
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")


# API Client Functions
def check_backend_health() -> Dict[str, Any]:
    """Check if FastAPI backend is healthy"""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Cannot connect to backend: {str(e)}")
        return None


def process_urls_api(urls: List[str]) -> Dict[str, Any]:
    """Process URLs via FastAPI backend"""
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/api/v1/process-urls",
            json={"urls": urls},
            timeout=300  # 5 minutes timeout for processing
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


def process_urls_async_api(urls: List[str]) -> Dict[str, Any]:
    """Process URLs asynchronously via FastAPI backend"""
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/api/v1/process-urls-async",
            json={"urls": urls},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


def query_api(question: str, include_sources: bool = True) -> Dict[str, Any]:
    """Query the knowledge base via FastAPI backend"""
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/api/v1/query",
            json={
                "question": question,
                "include_sources": include_sources
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


def clear_index_api() -> Dict[str, Any]:
    """Clear the vector store index via FastAPI backend"""
    try:
        response = requests.delete(
            f"{FASTAPI_BASE_URL}/api/v1/clear-index",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


# Initialize session state
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = 'synchronous'


# Sidebar: Backend health check
with st.sidebar:
    st.markdown("### Backend Status")
    health_status = check_backend_health()
    
    if health_status:
        status_color = "🟢" if health_status.get("status") == "healthy" else "🟡"
        st.markdown(f"{status_color} **Status:** {health_status.get('status', 'unknown').title()}")
        
        vectorstore_ready = health_status.get("vectorstore_ready", False)
        vector_status = "🟢 Ready" if vectorstore_ready else "🔴 Not Initialized"
        st.markdown(f"**Vector Store:** {vector_status}")
    else:
        st.markdown("🔴 **Status:** Backend Unavailable")
        st.stop()
    
    st.markdown("---")


# Sidebar: URL input and processing
st.sidebar.markdown("### Document Processing")

# Processing mode selection
processing_mode = st.sidebar.radio(
    "Processing Mode:",
    ["Synchronous", "Asynchronous (Background)"],
    help="Synchronous: Wait for completion. Asynchronous: Process in background."
)
st.session_state.processing_mode = processing_mode.lower().split()[0]

# URL inputs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs", type="primary", use_container_width=True)

main_placeholder = st.empty()


# Process URLs
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL")
    else:
        try:
            if st.session_state.processing_mode == 'synchronous':
                # Synchronous processing
                with st.spinner("Processing URLs..."):
                    main_placeholder.text("📥 Loading data from URLs...")
                    time.sleep(0.5)
                    
                    main_placeholder.text("🔄 Creating embeddings...")
                    time.sleep(0.5)
                    
                    result = process_urls_api(urls)
                    
                    main_placeholder.text("💾 Saving vector database...")
                    time.sleep(0.5)
                
                # Show results
                if result.get("status") == "success":
                    main_placeholder.success(
                        f"✅ {result.get('message')} - "
                        f"Processed {result.get('documents_processed')} document chunks"
                    )
                    st.sidebar.success(
                        f"✅ Successfully processed {result.get('documents_processed')} chunks"
                    )
                else:
                    st.error(f"Processing failed: {result}")
            
            else:
                # Asynchronous processing
                result = process_urls_async_api(urls)
                
                if result.get("status") == "processing":
                    st.sidebar.info(
                        f"⏳ {result.get('message')}\n\n"
                        "Processing in background. You can continue using the app."
                    )
                    main_placeholder.info(
                        "🔄 URLs submitted for background processing. "
                        "Check the backend status above to see when processing is complete."
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing URLs: {str(e)}")
            st.sidebar.error("Processing failed")


# Main content area
st.markdown("---")
st.markdown("### Ask Questions About Your Documents")

# Query interface
query = st.text_input(
    "Question:",
    placeholder="Enter your question here...",
    help="Ask anything about the processed documents"
)

# Query options
col1, col2 = st.columns([3, 1])
with col2:
    include_sources = st.checkbox("Show sources", value=True)

if query:
    # Check if vector store is ready
    health = check_backend_health()
    if not health or not health.get("vectorstore_ready"):
        st.warning("⚠️ Vector store not initialized. Please process URLs first.")
    else:
        try:
            with st.spinner("🤔 Thinking..."):
                result = query_api(query, include_sources=include_sources)
            
            # Display answer
            st.markdown("### Answer")
            st.markdown(f"**{result.get('answer')}**")
            
            # Display sources if requested
            if include_sources and result.get('sources'):
                st.markdown("---")
                with st.expander("📚 View Source Documents", expanded=False):
                    for i, source in enumerate(result['sources']):
                        st.markdown(f"#### Source {i+1}")
                        st.text_area(
                            f"Content {i+1}",
                            value=source['content'],
                            height=150,
                            key=f"source_{i}",
                            disabled=True
                        )
                        
                        if source.get('metadata'):
                            with st.expander(f"Metadata for Source {i+1}"):
                                st.json(source['metadata'])
                        
                        if i < len(result['sources']) - 1:
                            st.markdown("---")
                            
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")


# Sidebar: Additional options
st.sidebar.markdown("---")
st.sidebar.markdown("### Advanced Options")

if st.sidebar.button("🗑️ Clear Vector Database", use_container_width=True):
    try:
        with st.spinner("Clearing database..."):
            result = clear_index_api()
        
        if result.get("status") == "success":
            st.sidebar.success("✅ " + result.get("message"))
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error(f"Failed to clear: {result}")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <small>RockyBot RAG Application | Powered by FastAPI + Streamlit + Gemini + FAISS</small>
    </div>
    """,
    unsafe_allow_html=True
)