# AI Assistant Pro: Multi-Agent Tool-Calling System 🤖

[![LAMA Benchmark](https://img.shields.io/badge/LAMA-100%25-brightgreen)](https://github.com/mohammmadirfan/agentic-ai-bootcamp)
[![GSM8k Benchmark](https://img.shields.io/badge/GSM8k-90%25-blue)](https://github.com/mohammmadirfan/agentic-ai-bootcamp)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://github.com/mohammmadirfan/agentic-ai-bootcamp)

## 🎯 Overview
An intelligent AI agent system designed for task decomposition and dynamic tool selection. The system intelligently routes queries to specialized tools and chains them when needed for complex multi-step reasoning.

**Key Achievements:**
- ✅ **100% LAMA benchmark** (factual recall)
- ✅ **90% GSM8k benchmark** (math reasoning)
- ✅ **Sub-2-second latency** for RAG retrieval
- ✅ **4 integrated tools** with dynamic orchestration

## 🏗️ Architecture

The system uses a **Controller-Tool** architecture powered by **LangChain** and **Groq's Llama3 models**:
```
User Query → Agentic Controller → Tool Selection/Chaining → Response
```

### Agentic Controller
- **LLM:** Groq Llama3-70B/8B
- **Framework:** LangChain
- **Capabilities:** Task decomposition, tool selection, tool chaining

### Tool Suite

| Tool | Purpose | Technology |
|------|---------|------------|
| 🌐 **Web Search** | Real-time information retrieval | Serper API |
| 🧮 **Calculator** | Arithmetic & symbolic math | Sympy + Regex |
| ➗ **Math Solver** | Word problems & reasoning | Llama3-70B (Groq) |
| 📄 **Document QA** | RAG-based Q&A from docs | FAISS + HuggingFace |

## ✨ Key Features

### 1. Intelligent Task Routing
The controller analyzes queries and decides:
- **DIRECT:** Answer from LLM knowledge
- **TOOL:** Route to single specialized tool
- **CHAIN:** Chain multiple tools for complex tasks

### 2. Robust Prompt Engineering
- Distinguishes arithmetic vs. word problems
- Prioritizes RAG for private/document-based data
- Handles tool chaining for multi-step queries

### 3. Streamlit UI
- 🎨 Light/dark theme support
- 💬 Chat history with context
- 📊 Tool usage visualization (Plotly)
- 📁 Document upload for RAG

### 4. Performance Optimization
- Response caching
- Error handling & fallbacks
- Sub-2-second RAG latency

## 📊 Benchmark Results

### GSM8k (Mathematical Reasoning)
- **Score:** 90% (9/10 correct)
- **Task:** Grade school math word problems
- **Example:** *"A farmer has 15 cows. All but 8 die. How many are left?"*

### LAMA (Factual Recall)
- **Score:** 100% (10/10 correct)
- **Task:** Knowledge-based factual questions
- **Example:** *"Who wrote Romeo and Juliet?"*

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/mohammmadirfan/agentic-ai-bootcamp.git
cd agentic-ai-bootcamp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
```

### Run Application
```bash
streamlit run app.py
```
Access at: **http://localhost:8501**

## 💡 Example Queries

| Query Type | Example | Tool Used |
|------------|---------|-----------|
| Web Search | *"What's the latest news in Tokyo?"* | 🌐 Web Search |
| Arithmetic | *"Calculate 10 + 40"* | 🧮 Calculator |
| Word Problem | *"A farmer has 15 cows. All but 8 die. How many left?"* | ➗ Math Solver |
| Document QA | *"What's in the company handbook?"* | 📄 Document QA (RAG) |
| Direct Answer | *"Who wrote Romeo and Juliet?"* | 🤖 LLM Direct |

## 📂 Project Structure
```
agentic-ai-bootcamp/
├── agent/
│   ├── controller.py          # Main agent controller
│   └── tools/
│       ├── web_search.py      # Serper API integration
│       ├── calculator.py      # Sympy calculator
│       ├── math_solver.py     # LLM-based solver
│       └── document_qa.py     # RAG implementation
├── data/
│   ├── documents/             # RAG knowledge base
│   ├── benchmarks/            # LAMA & GSM8k datasets
│   └── results/               # Evaluation results
├── evaluation/
│   ├── evaluate_lama.py       # LAMA benchmark
│   └── evaluate_gsm8k.py      # GSM8k benchmark
├── app.py                     # Streamlit UI
├── requirements.txt
└── README.md
```

## 🧠 Agent Decision Flow
```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Controller       │
│ (Prompt Analysis)│
└──────┬───────────┘
       │
       ├──► DIRECT ──────► LLM Response
       │
       ├──► TOOL ────────► Select Tool ──► Execute ──► Response
       │
       └──► CHAIN ───────► Tool 1 ──► Tool 2 ──► Response
```

## 🛠️ Tech Stack
- **LLM:** Groq (Llama3-70B/8B)
- **Framework:** LangChain
- **Vector DB:** FAISS
- **Embeddings:** HuggingFace
- **UI:** Streamlit
- **APIs:** Serper (search), Groq (LLM)
- **Math Engine:** Sympy

## 📈 Performance Metrics
- **LAMA Accuracy:** 100% ✅
- **GSM8k Accuracy:** 90% ✅
- **RAG Latency:** < 2 seconds
- **Tool Selection Accuracy:** ~95%

## 🤝 Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-tool`
3. Commit changes: `git commit -m "Add new tool"`
4. Push and open Pull Request

## 📜 License
MIT License - see LICENSE file for details.

---

**100% LAMA • 90% GSM8k • Sub-2s Latency**