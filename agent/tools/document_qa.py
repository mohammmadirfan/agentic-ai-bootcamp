import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Document processing imports
import PyPDF2
import docx
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class DocumentQATool:
    """Document Question-Answering tool using RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self):
        """Initialize the Document QA tool"""
        self.description = "Answer questions based on uploaded documents using RAG"
        self.capabilities = [
            "Document summarization",
            "Information extraction",
            "Cross-document analysis",
            "Contextual question answering",
            "Knowledge base queries"
        ]
        
        # Initialize components
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = None
        self.llm = None
        self.documents_dir = Path("data/documents")
        self.vector_store_dir = Path("data/vector_store")
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components"""
        try:
            # Initialize embeddings (using a lightweight model)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Initialize LLM
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key:
                self.llm = ChatGroq(
                    model="llama3-70b-8192",
                    api_key=groq_api_key,
                    temperature=0.2
                )
            
            logger.info("Document QA components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Document QA components: {e}")
    
    def initialize_document_qa(self):
        """Initialize or reinitialize the document QA system with current documents"""
        try:
            # Create directories if they don't exist
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
            
            # Load all documents
            documents = self._load_all_documents()
            
            if not documents:
                logger.info("No documents found to process")
                return
            
            # Split documents into chunks
            doc_chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(doc_chunks)} chunks")
            
            # Create or update vector store
            if doc_chunks:
                self.vector_store = FAISS.from_documents(
                    documents=doc_chunks,
                    embedding=self.embeddings
                )
                
                # Save vector store
                vector_store_path = self.vector_store_dir / "faiss_index"
                self.vector_store.save_local(str(vector_store_path))
                
                logger.info("Vector store created and saved successfully")
            
        except Exception as e:
            logger.error(f"Error initializing document QA: {e}")
            raise
    
    def _load_all_documents(self) -> List[Document]:
        """Load all documents from the documents directory"""
        documents = []
        
        if not self.documents_dir.exists():
            return documents
        
        # Process each file in the documents directory
        for file_path in self.documents_dir.iterdir():
            if file_path.is_file():
                try:
                    docs = self._load_single_document(file_path)
                    documents.extend(docs)
                    logger.info(f"Loaded document: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {e}")
        
        return documents
    
    def _load_single_document(self, file_path: Path) -> List[Document]:
        """Load a single document based on its file type"""
        documents = []
        
        try:
            if file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path))
                documents = loader.load()
                
            elif file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    'source_file': file_path.name,
                    'file_type': file_path.suffix.lower(),
                    'loaded_at': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            # Create a simple document with error info
            documents = [Document(
                page_content=f"Error loading {file_path.name}: {str(e)}",
                metadata={'source_file': file_path.name, 'error': True}
            )]
        
        return documents
    
    def answer_question(self, question: str, top_k: int = 4) -> str:
        """
        Answer a question based on the document knowledge base
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Answer based on document content
        """
        try:
            # Check if vector store exists
            if not self.vector_store:
                # Try to load existing vector store
                vector_store_path = self.vector_store_dir / "faiss_index"
                if vector_store_path.exists():
                    try:
                        self.vector_store = FAISS.load_local(
                            str(vector_store_path),
                            embeddings=self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        logger.info("Loaded existing vector store")
                    except Exception as e:
                        logger.error(f"Error loading vector store: {e}")
                        return "❌ No documents available. Please upload documents first."
                else:
                    return "❌ No documents available. Please upload documents first."
            
            # Perform similarity search
            relevant_docs = self.vector_store.similarity_search(
                question, 
                k=top_k
            )
            
            if not relevant_docs:
                return "❌ No relevant information found in the documents."
            
            # Create context from relevant documents
            context = self._create_context(relevant_docs)
            
            # Generate answer using LLM
            if self.llm:
                answer = self._generate_answer(question, context)
            else:
                answer = self._fallback_answer(question, relevant_docs)
            
            return self._format_qa_response(question, answer, relevant_docs)
            
        except Exception as e:
            logger.error(f"Document QA error: {e}")
            return f"❌ Error processing your question: {str(e)}"
    
    def _create_context(self, relevant_docs: List[Document]) -> str:
        """Create context string from relevant documents"""
        context_parts = []
        
        for i, doc in enumerate(relevant_docs, 1):
            source_file = doc.metadata.get('source_file', 'Unknown')
            content = doc.page_content.strip()
            
            context_parts.append(f"Document {i} ({source_file}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context"""
        try:
            system_prompt = """You are a helpful document analysis assistant. Answer questions based ONLY on the provided document context.

Rules:
1. Use only information from the provided documents
2. If the answer isn't in the documents, say so clearly
3. Cite which document(s) you're referencing
4. Be concise but comprehensive
5. If multiple documents contain relevant info, synthesize them
6. Maintain accuracy - don't make assumptions beyond the text"""

            user_prompt = f"""
Context from documents:
{context}

Question: {question}

Please provide a detailed answer based on the document content above.
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._fallback_answer(question, context)
    
    def _fallback_answer(self, question: str, relevant_docs: List[Document]) -> str:
        """Fallback answer when LLM is not available"""
        answer = "Based on the uploaded documents:\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            source_file = doc.metadata.get('source_file', 'Unknown')
            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            answer += f"**From {source_file}:**\n{content}\n\n"
        
        answer += "*Note: Advanced analysis unavailable (LLM not configured)*"
        return answer
    
    def _format_qa_response(self, question: str, answer: str, sources: List[Document]) -> str:
        """Format the QA response with sources and metadata"""
        try:
            formatted_response = f"📄 **Document-Based Answer**\n\n"
            formatted_response += f"**Question:** {question}\n\n"
            formatted_response += f"**Answer:**\n{answer}\n\n"
            
            # Add source information
            formatted_response += "**Sources:**\n"
            unique_sources = set()
            for doc in sources:
                source_file = doc.metadata.get('source_file', 'Unknown')
                unique_sources.add(source_file)
            
            for source in sorted(unique_sources):
                formatted_response += f"• {source}\n"
            
            formatted_response += f"\n*Answer generated from {len(sources)} relevant document sections*"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Response formatting error: {e}")
            return f"📄 **Document-Based Answer**\n\n{answer}"
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of available documents and their content"""
        try:
            summary = {
                "total_documents": 0,
                "document_types": {},
                "total_chunks": 0,
                "documents": []
            }
            
            if not self.documents_dir.exists():
                return summary
            
            for file_path in self.documents_dir.iterdir():
                if file_path.is_file():
                    file_info = {
                        "name": file_path.name,
                        "type": file_path.suffix.lower(),
                        "size": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                    
                    summary["documents"].append(file_info)
                    summary["total_documents"] += 1
                    
                    file_type = file_path.suffix.lower()
                    summary["document_types"][file_type] = summary["document_types"].get(file_type, 0) + 1
            
            # Get chunk count from vector store if available
            if self.vector_store:
                summary["total_chunks"] = self.vector_store.index.ntotal
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting document summary: {e}")
            return {"error": str(e)}
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search documents for relevant content"""
        try:
            if not self.vector_store:
                return []
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get('source_file', 'Unknown'),
                    "score": float(score),
                    "metadata": doc.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

class DocumentQA:
    """Main Document QA class for integration with Streamlit app"""
    
    def __init__(self):
        """Initialize Document QA system"""
        self.tool = DocumentQATool()
    
    def initialize_document_qa(self):
        """Initialize the document QA system"""
        self.tool.initialize_document_qa()
    
    def answer_question(self, question: str) -> str:
        """Answer a question based on documents"""
        return self.tool.answer_question(question)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get document summary"""
        return self.tool.get_document_summary()
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search documents"""
        return self.tool.search_documents(query)