import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import sys
import logging
import datetime
import json
import shutil
from typing import Dict, List, Any

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

# Import our custom modules
from agent.controller import AgentController
from agent.tools.document_qa import DocumentQA
from evaluation.evaluate_lama import run_lama_evaluation
from evaluation.evaluate_gsm8k import run_gsm8k_evaluation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Agentic AI Bootcamp Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --background-dark: #0f172a;
        --background-light: #f8fafc;
        --surface-dark: rgba(30, 41, 59, 0.8);
        --surface-light: rgba(255, 255, 255, 0.8);
        --text-primary-dark: #f1f5f9;
        --text-primary-light: #1e293b;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        min-height: 100vh;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        margin: 0;
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--accent-color);
    }
    
    .chat-message.user {
        background: rgba(99, 102, 241, 0.2);
        border-left-color: var(--primary-color);
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background: rgba(139, 92, 246, 0.2);
        border-left-color: var(--secondary-color);
        margin-right: 2rem;
    }
    
    .tool-badge {
        display: inline-block;
        background: var(--accent-color);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .metric-card h3 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: rgba(255, 255, 255, 0.8);
        margin: 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
    }
    
    .sidebar .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .evaluation-result {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid var(--success-color);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .document-item {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--accent-color);
    }
    
    /* Dark theme adjustments */
    [data-theme="dark"] {
        --bg-color: var(--background-dark);
        --surface-color: var(--surface-dark);
        --text-color: var(--text-primary-dark);
    }
    
    /* Light theme adjustments */
    [data-theme="light"] {
        --bg-color: var(--background-light);
        --surface-color: var(--surface-light);
        --text-color: var(--text-primary-light);
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    /* Animation classes */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'past_history' not in st.session_state:
        st.session_state.past_history = []
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "💬 Chat"
    
    if 'agent_controller' not in st.session_state:
        st.session_state.agent_controller = AgentController()
    
    if 'document_qa' not in st.session_state:
        st.session_state.document_qa = DocumentQA()
    
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}
    
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"

# Main header component
def render_main_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header slide-in">
        <h1>🤖 Agentic AI Bootcamp Hub</h1>
        <p>Advanced Multi-Tool AI Agent with Dynamic Routing & Performance Analytics</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        pages = [
            "💬 Chat",
            "📊 Evaluation", 
            "📁 Documents",
            "ℹ️ About"
        ]
        
        selected_page = st.selectbox(
            "Choose a page:",
            pages,
            index=pages.index(st.session_state.current_page),
            key="page_selector"
        )
        
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()
        
        st.markdown("---")
        
        # Theme toggle
        theme_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.theme == "dark")
        if theme_toggle != (st.session_state.theme == "dark"):
            st.session_state.theme = "dark" if theme_toggle else "light"
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chats", len(st.session_state.past_history))
        with col2:
            st.metric("Docs", len(st.session_state.uploaded_documents))

# Chat page
def render_chat_page():
    """Render the main chat interface"""
    st.markdown("## 💬 Chat with AI Agent")
    
    # Quick example buttons
    st.markdown("### 🚀 Quick Examples")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Web Search", help="Search for current information"):
            example_query = "What are the latest developments in AI?"
            st.session_state.example_query = example_query
    
    with col2:
        if st.button("🧮 Calculator", help="Solve math problems"):
            example_query = "What is the square root of 144 plus 25?"
            st.session_state.example_query = example_query
    
    with col3:
        if st.button("📄 Document QA", help="Ask about uploaded documents"):
            example_query = "Summarize the key points from my documents"
            st.session_state.example_query = example_query
    
    with col4:
        if st.button("💭 General Chat", help="General conversation"):
            example_query = "Explain the concept of machine learning"
            st.session_state.example_query = example_query
    
    # Chat input
    chat_input = st.text_area(
        "Your message:",
        value=getattr(st.session_state, 'example_query', ''),
        height=100,
        placeholder="Ask me anything! I can search the web, solve math, analyze documents, or chat..."
    )
    
    if 'example_query' in st.session_state:
        del st.session_state.example_query
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("Send 🚀", type="primary")
    with col2:
        if st.button("Clear History 🗑️"):
            st.session_state.past_history = []
            st.rerun()
    
    # Process chat input
    if send_button and chat_input.strip():
        with st.spinner("🤔 Agent is thinking..."):
            try:
                response = st.session_state.agent_controller.process_query(chat_input)
                
                # Add to history
                st.session_state.past_history.append({
                    'query': chat_input,
                    'response': response['response'],
                    'tool_used': response['tool_used'],
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Chat error: {e}")
    
    # Display chat history
    st.markdown("### 📝 Chat History")
    
    if st.session_state.past_history:
        chat_container = st.container()
        with chat_container:
            for i, exchange in enumerate(reversed(st.session_state.past_history)):
                # User message
                st.markdown(f"""
                <div class="chat-message user">
                    <strong>You:</strong><br>
                    {exchange['query']}
                </div>
                """, unsafe_allow_html=True)
                
                # Assistant message with tool badge
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="tool-badge">{exchange['tool_used']}</div>
                    <strong>Assistant:</strong><br>
                    {exchange['response']}
                </div>
                """, unsafe_allow_html=True)
                
                if i < len(st.session_state.past_history) - 1:
                    st.markdown("---")
    else:
        st.info("🌟 Start a conversation! The AI agent will automatically choose the best tool for your query.")

# Evaluation page
def render_evaluation_page():
    """Render the evaluation and benchmarking page"""
    st.markdown("## 📊 Agent Evaluation & Benchmarks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 LAMA Knowledge Test")
        st.write("Test the agent's factual knowledge and reasoning capabilities")
        
        if st.button("Run LAMA Evaluation", type="primary"):
            with st.spinner("Running LAMA evaluation..."):
                try:
                    results = run_lama_evaluation(st.session_state.agent_controller)
                    st.session_state.evaluation_results['lama'] = results
                    st.success("LAMA evaluation completed!")
                except Exception as e:
                    st.error(f"LAMA evaluation failed: {str(e)}")
    
    with col2:
        st.markdown("### 🔢 GSM8K Math Test")
        st.write("Evaluate mathematical reasoning and problem-solving skills")
        
        if st.button("Run GSM8K Evaluation", type="primary"):
            with st.spinner("Running GSM8K evaluation..."):
                try:
                    results = run_gsm8k_evaluation(st.session_state.agent_controller)
                    st.session_state.evaluation_results['gsm8k'] = results
                    st.success("GSM8K evaluation completed!")
                except Exception as e:
                    st.error(f"GSM8K evaluation failed: {str(e)}")
    
    # Display results
    if st.session_state.evaluation_results:
        st.markdown("### 📈 Results")
        
        # Create metrics display
        if 'lama' in st.session_state.evaluation_results:
            lama_score = st.session_state.evaluation_results['lama'].get('accuracy', 0)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{lama_score:.1%}</h3>
                    <p>LAMA Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
        
        if 'gsm8k' in st.session_state.evaluation_results:
            gsm8k_score = st.session_state.evaluation_results['gsm8k'].get('accuracy', 0)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{gsm8k_score:.1%}</h3>
                    <p>GSM8K Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance chart
        if len(st.session_state.evaluation_results) > 0:
            chart_data = []
            colors = []
            
            for test_name, results in st.session_state.evaluation_results.items():
                score = results.get('accuracy', 0)
                chart_data.append({'Test': test_name.upper(), 'Accuracy': score * 100})
                
                # Color coding based on performance
                if score >= 0.8:
                    colors.append('#10b981')  # Green
                elif score >= 0.6:
                    colors.append('#f59e0b')  # Yellow
                else:
                    colors.append('#ef4444')  # Red
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=df['Test'],
                        y=df['Accuracy'],
                        marker_color=colors,
                        text=df['Accuracy'].apply(lambda x: f'{x:.1f}%'),
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Agent Performance Benchmarks",
                    xaxis_title="Benchmark Test",
                    yaxis_title="Accuracy (%)",
                    yaxis=dict(range=[0, 100]),
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Documents page
def render_documents_page():
    """Render the document management page"""
    st.markdown("## 📁 Document Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload TXT, PDF, or DOCX files to create your knowledge base"
        )
        
        if uploaded_files:
            if st.button("Process Files", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        # Create documents directory if it doesn't exist
                        doc_dir = Path("data/documents")
                        doc_dir.mkdir(parents=True, exist_ok=True)
                        
                        processed_files = []
                        for file in uploaded_files:
                            # Save file
                            file_path = doc_dir / file.name
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                            
                            processed_files.append({
                                'name': file.name,
                                'size': file.size,
                                'type': file.type,
                                'upload_time': datetime.datetime.now().isoformat()
                            })
                        
                        # Update document list
                        st.session_state.uploaded_documents.extend(processed_files)
                        
                        # Initialize document QA
                        st.session_state.document_qa.initialize_document_qa()
                        
                        st.success(f"Successfully processed {len(processed_files)} files!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
                        logger.error(f"Document processing error: {e}")
    
    with col2:
        st.markdown("### 📊 Document Stats")
        
        total_docs = len(st.session_state.uploaded_documents)
        total_size = sum(doc.get('size', 0) for doc in st.session_state.uploaded_documents)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_docs}</h3>
            <p>Total Documents</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_size / 1024:.1f} KB</h3>
            <p>Total Size</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Document list
    if st.session_state.uploaded_documents:
        st.markdown("### 📋 Uploaded Documents")
        
        df = pd.DataFrame(st.session_state.uploaded_documents)
        st.dataframe(
            df[['name', 'type', 'size', 'upload_time']],
            use_container_width=True,
            hide_index=True
        )
        
        # Clear documents button
        if st.button("Clear All Documents 🗑️", type="secondary"):
            if st.confirm("Are you sure you want to delete all documents?"):
                st.session_state.uploaded_documents = []
                # Clear the documents directory
                doc_dir = Path("data/documents")
                if doc_dir.exists():
                    shutil.rmtree(doc_dir)
                st.success("All documents cleared!")
                st.rerun()

# About page
def render_about_page():
    """Render the about page with system information"""
    st.markdown("## ℹ️ About the AI Agent")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Key Features
        
        **🔧 Multi-Tool Agent**
        - Intelligent tool routing based on query analysis
        - Web search for real-time information
        - Advanced mathematical calculations
        - Document analysis with RAG
        
        **⚡ Performance Focused**
        - Built-in benchmarking with LAMA & GSM8K
        - Real-time performance monitoring
        - Optimization feedback loops
        
        **🎨 Modern Interface**
        - Glassmorphism design with smooth animations
        - Dark/Light theme support
        - Responsive layout for all devices
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Technical Stack
        
        **🧠 AI Framework**
        - LangGraph for agent orchestration
        - Multiple LLM models (Llama 3-8B, 3-70B)
        - Custom tool implementations
        
        **🎯 Tools & APIs**
        - Serper API for web search
        - Document processing (PDF, DOCX, TXT)
        - Mathematical computation engine
        
        **🖥️ Frontend**
        - Streamlit for rapid development
        - Custom CSS for modern aesthetics
        - Plotly for interactive visualizations
        """)
    
    st.markdown("### 📖 Usage Guide")
    
    with st.expander("🔍 How to Use Web Search"):
        st.write("""
        Simply ask questions that require current information:
        - "What's the latest news about AI?"
        - "Current weather in New York"
        - "Recent stock prices for AAPL"
        """)
    
    with st.expander("🧮 How to Use Calculator"):
        st.write("""
        Ask mathematical questions or calculations:
        - "What is 15% of 240?"
        - "Solve: 2x + 5 = 17"
        - "Calculate compound interest for $1000 at 5% for 3 years"
        """)
    
    with st.expander("📄 How to Use Document QA"):
        st.write("""
        Upload documents first, then ask questions:
        - "Summarize the main points"
        - "What does the document say about [topic]?"
        - "Find information related to [keyword]"
        """)

# Main application logic
def main():
    """Main application entry point"""
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Render main header
    render_main_header()
    
    # Render sidebar
    render_sidebar()
    
    # Render current page
    if st.session_state.current_page == "💬 Chat":
        render_chat_page()
    elif st.session_state.current_page == "📊 Evaluation":
        render_evaluation_page()
    elif st.session_state.current_page == "📁 Documents":
        render_documents_page()
    elif st.session_state.current_page == "ℹ️ About":
        render_about_page()

if __name__ == "__main__":
    main()