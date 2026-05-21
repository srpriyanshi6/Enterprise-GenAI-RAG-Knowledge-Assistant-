"""Beautiful Streamlit UI for RAG Assistant"""
import streamlit as st
import requests
import json
from typing import List, Dict
import time
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .answer-text {
        font-size: 16px;
        line-height: 1.6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

API_URL = "http://localhost:8000"

def upload_documents(files):
    """Upload documents to backend"""
    with st.spinner("Processing documents..."):
        files_data = [("files", (file.name, file, file.type)) for file in files]
        response = requests.post(
            f"{API_URL}/upload",
            files=files_data,
            params={"session_id": st.session_state.session_id}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data['session_id']
            st.session_state.documents_loaded = True
            return data
        else:
            st.error(f"Upload failed: {response.text}")
            return None

def query_documents(question: str, use_hybrid: bool):
    """Query the documents"""
    response = requests.post(
        f"{API_URL}/query",
        json={
            "question": question,
            "session_id": st.session_state.session_id,
            "use_hybrid": use_hybrid
        }
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Query failed: {response.text}")
        return None

with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("RAG Assistant")
    st.markdown("---")
    
    # Configuration
    st.subheader("Configuration")
    use_hybrid = st.toggle("Enable Hybrid Search", value=True)
    st.caption("Combines semantic + keyword search for better results")
    
    # Upload section
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing..."):
            result = upload_documents(uploaded_files)
            if result:
                st.success(f"Processed {result['documents_processed']} documents")
                st.info(f"Created {result['chunks_created']} text chunks")
    
    # Stats
    if st.session_state.documents_loaded:
        st.markdown("---")
        st.subheader("Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", len(st.session_state.query_history))
        with col2:
            avg_time = sum([q.get('processing_time_ms', 0) for q in st.session_state.query_history]) / max(len(st.session_state.query_history), 1)
            st.metric("Avg Response", f"{avg_time:.0f}ms")
    
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("""
    - Semantic Search
    - Hybrid Retrieval  
    - Source Citations
    - Multi-Document QA
    - Conversation Memory
    """)

st.title("Enterprise GenAI Knowledge Assistant")
st.markdown("*Retrieval-Augmented Generation with Hybrid Search*")

if not st.session_state.documents_loaded:
    st.info("**Welcome** Upload documents in the sidebar to start asking questions.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **What you can do:**
        - Upload PDF reports
        - Ask complex questions
        - Get cited answers
        - Export conversations
        """)
    with col2:
        st.markdown("""
        **Technical Features:**
        - Hybrid Search (BM25 + Vectors)
        - Context-aware responses
        - Hallucination reduction
        - Streaming responses
        """)
    with col3:
        st.markdown("""
        **Sample Questions:**
        - Summarize key findings
        - Compare across documents
        - Find specific details
        - Extract action items
        """)

if st.session_state.documents_loaded:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['source']}** (Page {source['page']})")
    
    if prompt := st.chat_input("Ask a question about your documents..."):
       
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        #response 
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_documents(prompt, use_hybrid)
                
                if response:
                    st.markdown(response['answer'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sources Found", len(response['sources']))
                    with col2:
                        st.metric("Response Time", f"{response['processing_time_ms']}ms")
                    with col3:
                        st.metric("Search Method", "Hybrid" if response['hybrid_used'] else "Semantic")
                    
                    if response['sources']:
                        with st.expander("Source Documents"):
                            for source in response['sources']:
                                st.markdown(f"""
                                <div class='source-card'>
                                    **{source['source']}** - Page {source['page']}<br>
                                    <small>Relevance Score: {source.get('score', 'N/A')}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['answer'],
                        "sources": response['sources']
                    })
                    st.session_state.query_history.append(response)
    
    if st.button("Export Conversation"):
        export_data = {
            "messages": st.session_state.messages,
            "metadata": {
                "session_id": st.session_state.session_id,
                "total_queries": len(st.session_state.query_history)
            }
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name="conversation_export.json",
            mime="application/json"
        )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with LangChain, FAISS, FastAPI, Streamlit | Hybrid Search RAG System
</div>
""", unsafe_allow_html=True)