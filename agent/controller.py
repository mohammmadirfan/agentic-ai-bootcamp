from typing import Dict, List, Any, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pathlib import Path
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import our custom tools
from .tools.web_search import WebSearchTool
from .tools.calculator import CalculatorTool
from .tools.math_solver import MathSolverTool
from .tools.document_qa import DocumentQATool

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: Annotated[List[Any], operator.add]
    query: str
    tool_used: str
    response: str
    routing_decision: str
    error: str

class AgentController:
    """LangGraph-based agent controller for intelligent tool routing"""
    
    def __init__(self):
        """Initialize the agent controller with LangGraph workflow"""
        self.setup_llm()
        self.setup_tools()
        self.build_graph()
    
    def setup_llm(self):
        """Setup the language models"""
        # Use environment variables for API keys
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Fast model for routing and general tasks
        self.llm_fast = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.1
        )
        
        # Powerful model for complex reasoning
        self.llm_powerful = ChatGroq(
            model="Llama-3.3-70B-Versatile",
            api_key=groq_api_key,
            temperature=0.3
        )
    
    def setup_tools(self):
        """Initialize all available tools"""
        self.tools = {
            'web_search': WebSearchTool(),
            'calculator': CalculatorTool(),
            'math_solver': MathSolverTool(),
            'document_qa': DocumentQATool()
        }
        
        # Create LangChain tools for LangGraph
        self.langchain_tools = []
        
        @tool
        def web_search_tool(query: str) -> str:
            """Search the web for current information"""
            try:
                return self.tools['web_search'].search(query)
            except Exception as e:
                return f"Web search error: {str(e)}"
        
        @tool
        def calculator_tool(expression: str) -> str:
            """Perform mathematical calculations"""
            try:
                return self.tools['calculator'].calculate(expression)
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        @tool  
        def math_solver_tool(problem: str) -> str:
            """Solve complex mathematical problems step by step"""
            try:
                return self.tools['math_solver'].solve(problem)
            except Exception as e:
                return f"Math solving error: {str(e)}"
        
        @tool
        def document_qa_tool(question: str) -> str:
            """Answer questions based on uploaded documents"""
            try:
                return self.tools['document_qa'].answer_question(question)
            except Exception as e:
                return f"Document QA error: {str(e)}"
        
        self.langchain_tools = [
            web_search_tool,
            calculator_tool,
            math_solver_tool,
            document_qa_tool
        ]
        
        # Create tool node for LangGraph
        self.tool_node = ToolNode(self.langchain_tools)
    
    def build_graph(self):
        """Build the LangGraph workflow"""
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.route_query)
        workflow.add_node("web_search", self.use_web_search)
        workflow.add_node("calculator", self.use_calculator)
        workflow.add_node("math_solver", self.use_math_solver)
        workflow.add_node("document_qa", self.use_document_qa)
        workflow.add_node("general_chat", self.use_general_chat)
        
        # Add edges with conditional routing
        workflow.add_edge(START, "router")
        
        # Conditional edges from router to appropriate tools
        workflow.add_conditional_edges(
            "router",
            self.determine_route,
            {
                "web_search": "web_search",
                "calculator": "calculator", 
                "math_solver": "math_solver",
                "document_qa": "document_qa",
                "general_chat": "general_chat"
            }
        )
        
        # All tool nodes end the workflow
        workflow.add_edge("web_search", END)
        workflow.add_edge("calculator", END)
        workflow.add_edge("math_solver", END)
        workflow.add_edge("document_qa", END)
        workflow.add_edge("general_chat", END)
        
        # Compile the graph
        self.app = workflow.compile()
    
    def route_query(self, state: AgentState) -> AgentState:
        """Analyze the query and determine which tool to use"""
        query = state["query"]
        
        routing_prompt = f"""You are an expert query router. Analyze this user query and select the single best tool.

        QUERY: "{query}"

        TOOLS & SELECTION CRITERIA:

        🔍 web_search - Select if query contains:
        - Time indicators: "today", "current", "latest", "recent", "now", "2024", "2025"
        - News/events: "news", "happened", "announced", "breaking"
        - Real-time data: "price", "weather", "stock", "score"
        - Verification needs: "fact check", "confirm", "verify"

        🧮 calculator - Select for:
        - Arithmetic expressions: "2+3", "15*4", "100/5"
        - Unit conversions: "miles to km", "celsius to fahrenheit"
        - Percentage calculations: "20% of 150"
        - Simple numeric operations (no variables/equations)

        🔢 math_solver - Select for:
        - Equations with variables: "solve for x", "find y when"
        - Advanced math: "derivative", "integral", "matrix", "logarithm"
        - Word problems: mathematical scenarios requiring multi-step solving
        - Step-by-step math explanations needed

        📄 document_qa - Select if query mentions:
        - Document references: "my document", "the file", "uploaded"
        - Specific content: "in the paper", "according to", "from the report"
        - Previously provided information context

        💭 general_chat - Default for:
        - Explanations, definitions, how-to questions
        - Creative tasks, brainstorming, advice
        - General knowledge not requiring real-time data
        - Conversations, opinions, recommendations

        DECISION RULES:
        1. If multiple tools could work, prioritize: web_search > math_solver > calculator > document_qa > general_chat
        2. When uncertain between calculator/math_solver: choose calculator for simple arithmetic, math_solver for complex problems
        3. Only choose document_qa if query explicitly references documents/files
        4. Default to general_chat when no other tool clearly fits

        OUTPUT: Return only the tool name (no explanations): web_search, calculator, math_solver, document_qa, or general_chat

        EXAMPLES:
        "What's the latest news about AI?" → web_search
        "Calculate 15% tip on $80" → calculator  
        "Solve: 2x + 5 = 15" → math_solver
        "What does my resume say about experience?" → document_qa
        "Explain quantum physics" → general_chat
        """
        
        try:
            response = self.llm_fast.invoke([SystemMessage(content=routing_prompt)])
            routing_decision = response.content.strip().lower()
            
            # Enhanced validation with fallback logic
            valid_routes = ["web_search", "calculator", "math_solver", "document_qa", "general_chat"]
            
            if routing_decision not in valid_routes:
                # Smart fallback based on query patterns
                routing_decision = self._fallback_routing(query)
            
            state["routing_decision"] = routing_decision
            logger.info(f"Query: '{query[:50]}...' → Routed to: {routing_decision}")
            
        except Exception as e:
            logger.error(f"Routing error: {e}")
            state["routing_decision"] = self._fallback_routing(query)
            state["error"] = f"Routing error: {str(e)}"
        
        return state

    def _fallback_routing(self, query: str) -> str:
        """Fallback routing logic using simple pattern matching"""
        query_lower = query.lower()
        
        # Time-sensitive indicators
        time_indicators = ["today", "current", "latest", "recent", "now", "2024", "2025", "news"]
        if any(indicator in query_lower for indicator in time_indicators):
            return "web_search"
        
        # Math calculation patterns
        math_operators = ["+", "-", "*", "/", "=", "%"]
        if any(op in query for op in math_operators):
            # Complex math indicators
            complex_math = ["solve", "equation", "derivative", "integral", "x=", "find x"]
            if any(term in query_lower for term in complex_math):
                return "math_solver"
            return "calculator"
        
        #     Document references
        doc_indicators = ["document", "file", "uploaded", "my resume", "the report", "according to"]
        if any(indicator in query_lower for indicator in doc_indicators):
            return "document_qa"
        
        return "general_chat"
    
    def determine_route(self, state: AgentState) -> str:
        """Return the routing decision for conditional edges"""
        return state.get("routing_decision", "general_chat")
    
    def use_web_search(self, state: AgentState) -> AgentState:
        """Execute web search tool"""
        try:
            result = self.tools['web_search'].search(state["query"])
            state["response"] = result
            state["tool_used"] = "🔍 Web Search"
        except Exception as e:
            state["response"] = f"Web search failed: {str(e)}"
            state["tool_used"] = "🔍 Web Search (Error)"
            state["error"] = str(e)
        
        return state
    
    def use_calculator(self, state: AgentState) -> AgentState:
        """Execute calculator tool"""
        try:
            result = self.tools['calculator'].calculate(state["query"])
            state["response"] = result
            state["tool_used"] = "🧮 Calculator"
        except Exception as e:
            state["response"] = f"Calculation failed: {str(e)}"
            state["tool_used"] = "🧮 Calculator (Error)"
            state["error"] = str(e)
        
        return state
    
    def use_math_solver(self, state: AgentState) -> AgentState:
        """Execute math solver tool"""
        try:
            result = self.tools['math_solver'].solve(state["query"])
            state["response"] = result
            state["tool_used"] = "🔢 Math Solver"
        except Exception as e:
            state["response"] = f"Math solving failed: {str(e)}"
            state["tool_used"] = "🔢 Math Solver (Error)"
            state["error"] = str(e)
        
        return state
    
    def use_document_qa(self, state: AgentState) -> AgentState:
        """Execute document QA tool"""
        try:
            result = self.tools['document_qa'].answer_question(state["query"])
            state["response"] = result
            state["tool_used"] = "📄 Document QA"
        except Exception as e:
            state["response"] = f"Document QA failed: {str(e)}"
            state["tool_used"] = "📄 Document QA (Error)"
            state["error"] = str(e)
        
        return state
    
    def use_general_chat(self, state: AgentState) -> AgentState:
        """Execute general chat using LLM"""
        try:
            messages = [
                SystemMessage(content="""You are a helpful AI assistant. Provide informative, accurate, and engaging responses. 
                Be conversational but professional. If you don't know something, say so clearly."""),
                HumanMessage(content=state["query"])
            ]
            
            response = self.llm_powerful.invoke(messages)
            state["response"] = response.content
            state["tool_used"] = "💭 General Chat"
            
        except Exception as e:
            state["response"] = f"General chat failed: {str(e)}"
            state["tool_used"] = "💭 General Chat (Error)"
            state["error"] = str(e)
        
        return state
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the LangGraph workflow"""
        try:
            # Initialize state
            initial_state = AgentState(
                messages=[],
                query=query,
                tool_used="",
                response="",
                routing_decision="",
                error=""
            )
            
            # Run the workflow
            final_state = self.app.invoke(initial_state)
            
            # Log the interaction
            self.log_interaction(query, final_state)
            
            return {
                "response": final_state["response"],
                "tool_used": final_state["tool_used"],
                "routing_decision": final_state["routing_decision"],
                "error": final_state.get("error", "")
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "tool_used": "❌ Error",
                "routing_decision": "error",
                "error": str(e)
            }
    
    def log_interaction(self, query: str, final_state: AgentState):
        """Log the interaction for analysis and debugging"""
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "routing_decision": final_state["routing_decision"],
                "tool_used": final_state["tool_used"],
                "response_length": len(final_state["response"]),
                "error": final_state.get("error", "")
            }
            
            # Ensure logs directory exists
            logs_dir = Path("data/logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to daily log file
            log_file = logs_dir / f"interactions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
                
        except Exception as e:
            logger.error(f"Logging error: {e}")
    
    def get_available_tools(self) -> List[str]:
        """Return list of available tools"""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool"""
        if tool_name in self.tools:
            return {
                "name": tool_name,
                "description": getattr(self.tools[tool_name], 'description', 'No description available'),
                "capabilities": getattr(self.tools[tool_name], 'capabilities', [])
            }
        return {"error": "Tool not found"}
    
    def get_routing_stats(self) -> Dict[str, int]:
        """Get statistics about tool routing decisions"""
        try:
            logs_dir = Path("data/logs")
            if not logs_dir.exists():
                return {}
            
            stats = {}
            for log_file in logs_dir.glob("interactions_*.jsonl"):
                with open(log_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            tool = data.get("routing_decision", "unknown")
                            stats[tool] = stats.get(tool, 0) + 1
                        except json.JSONDecodeError:
                            continue
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}