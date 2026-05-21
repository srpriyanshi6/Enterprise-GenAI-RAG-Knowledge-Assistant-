"""Advanced RAG Engine with Hybrid Search and Re-ranking"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
from pathlib import Path

from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

load_dotenv()

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    content: str
    source: str
    page: int
    score: float
    relevance_score: float

class HybridRetriever:
    """Combines semantic search with keyword-based BM25"""
    
    def __init__(self, vector_store, documents: List[Document]):
        self.vector_store = vector_store
        self.documents = documents
        tokenized_docs = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def hybrid_search(self, query: str, k: int = 4, alpha: float = 0.5) -> List[SearchResult]:
       
        semantic_results = self.vector_store.similarity_search_with_score(query, k=k*2)
        
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:k*2]
        
        # combine scores
        #weighted
        combined_results = {}
        
        for doc, score in semantic_results:
            idx = self.documents.index(doc)
            combined_results[idx] = {
                'doc': doc,
                'score': alpha * (1 - score/2),  # Normalize semantic score
                'bm25_score': 0
            }
            
        for idx in top_bm25_idx:
            if idx in combined_results:
                combined_results[idx]['bm25_score'] = (1-alpha) * bm25_scores[idx]
                combined_results[idx]['score'] += combined_results[idx]['bm25_score']
            else:
                combined_results[idx] = {
                    'doc': self.documents[idx],
                    'score': (1-alpha) * bm25_scores[idx],
                    'bm25_score': (1-alpha) * bm25_scores[idx]
                }
        
        # sort and return top k
        sorted_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)[:k]
        
        return [
            SearchResult(
                content=res['doc'].page_content,
                source=res['doc'].metadata.get('source', 'Unknown'),
                page=res['doc'].metadata.get('page', 0),
                score=res['score'],
                relevance_score=res['score']
            )
            for res in sorted_results
        ]

class RAGEngine:
    """Main RAG Engine with advanced features"""
    
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.documents = []
        
        if use_openai:
            self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        self.prompt_template = PromptTemplate(
            template="""You are an expert enterprise assistant. Answer based ONLY on the provided context.
            ALWAYS cite your sources using [Source: filename, Page: X] format.
            
            Context from documents:
            {context}
            
            Question: {question}
            
            Instructions:
            1. If unsure, say "I cannot find this information in the provided documents"
            2. Never make up information
            3. Provide specific citations for each claim
            4. Be concise but thorough
            
            Answer: """,
            input_variables=["context", "question"]
        )
    
    def load_documents(self, file_paths: List[str]) -> int:
        """Load and chunk documents"""
        documents = []
        
        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                continue
                
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = Path(file_path).name
            documents.extend(docs)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.documents = text_splitter.split_documents(documents)
        return len(self.documents)
    
    def create_vector_store(self, persist_dir: str = "./vector_store"):
        """Create vector store from documents"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        self.vector_store = FAISS.from_documents(
            self.documents, 
            self.embeddings
        )
        
        self.retriever = HybridRetriever(self.vector_store, self.documents)
        
        if self.use_openai:
            llm = OpenAI(temperature=0.3, max_tokens=500)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            model_name = os.getenv("LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
            llm = HuggingFacePipeline(pipeline=pipe)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": int(os.getenv("TOP_K_RESULTS", 4))}
            ),
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
        
        return True
    
    def query(self, question: str, use_hybrid: bool = True) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.qa_chain:
            raise ValueError("Vector store not created. Call create_vector_store first.")
        
        if use_hybrid and self.retriever:
            results = self.retriever.hybrid_search(
                question, 
                k=int(os.getenv("TOP_K_RESULTS", 4))
            )
            
            context = "\n\n".join([
                f"[Source: {r.source}, Page: {r.page}]\n{r.content}"
                for r in results
            ])
            
            #answer
            response = self.qa_chain.llm_chain.predict(
                question=question,
                context=context
            )
            
            return {
                "answer": response,
                "sources": [{"source": r.source, "page": r.page, "score": r.score} 
                           for r in results],
                "hybrid_search_used": True
            }
        else:
            result = self.qa_chain({"query": question})
            sources = [
                {"source": doc.metadata.get('source', 'Unknown'), 
                 "page": doc.metadata.get('page', 0)}
                for doc in result['source_documents']
            ]
            
            return {
                "answer": result['result'],
                "sources": sources,
                "hybrid_search_used": False
            }