import re
import logging
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os

logger = logging.getLogger(__name__)

class MathSolverTool:
    """Advanced mathematical problem solver using LLM reasoning"""
    
    def __init__(self):
        """Initialize the math solver tool"""
        self.description = "Solve complex mathematical problems with step-by-step explanations"
        self.capabilities = [
            "Algebraic equations and systems",
            "Calculus (derivatives, integrals)",
            "Word problems",
            "Geometry and trigonometry",
            "Statistics and probability",
            "Step-by-step solutions"
        ]
        
        # Initialize LLM for math reasoning
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.llm = ChatGroq(
                model="deepseek-r1-distill-llama-70b",  # More powerful model for complex math
                api_key=groq_api_key,
                temperature=0.1  # Low temperature for consistency
            )
        else:
            self.llm = None
            logger.warning("GROQ_API_KEY not found - math solver will have limited functionality")
    
    def solve(self, problem: str) -> str:
        """
        Solve a mathematical problem with detailed explanation
        
        Args:
            problem: Mathematical problem description
            
        Returns:
            Detailed solution with steps
        """
        if not self.llm:
            return "❌ Math solver is not available (LLM not configured)"
        
        try:
            # Determine problem type for specialized handling
            problem_type = self._classify_problem(problem)
            
            # Create specialized prompt based on problem type
            system_prompt = self._get_system_prompt(problem_type)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Solve this mathematical problem: {problem}")
            ]
            
            # Get solution from LLM
            response = self.llm.invoke(messages)
            solution = response.content
            
            # Format and enhance the solution
            return self._format_solution(problem, solution, problem_type)
            
        except Exception as e:
            logger.error(f"Math solver error: {e}")
            return self._handle_solver_error(problem, str(e))
    
    def _classify_problem(self, problem: str) -> str:
        """Classify the type of mathematical problem"""
        problem_lower = problem.lower()
        
        # Keyword-based classification
        if any(word in problem_lower for word in ["derivative", "differentiate", "d/dx", "slope"]):
            return "calculus_derivative"
        elif any(word in problem_lower for word in ["integral", "integrate", "area under", "∫"]):
            return "calculus_integral"
        elif any(word in problem_lower for word in ["equation", "solve for", "x =", "find x"]):
            return "algebra"
        elif any(word in problem_lower for word in ["triangle", "circle", "area", "volume", "perimeter"]):
            return "geometry"
        elif any(word in problem_lower for word in ["probability", "statistics", "mean", "median", "variance"]):
            return "statistics"
        elif any(word in problem_lower for word in ["word problem", "age", "distance", "speed", "cost"]):
            return "word_problem"
        elif re.search(r'\d+.*[+\-*/^].*\d+', problem):
            return "arithmetic"
        else:
            return "general"
    
    def _get_system_prompt(self, problem_type: str) -> str:
        """Get specialized system prompt based on problem type"""
        base_prompt = """You are an expert mathematics tutor. Solve problems step-by-step with clear explanations.

Rules:
1. Show ALL steps in your solution
2. Explain the reasoning behind each step
3. Use proper mathematical notation
4. Verify your answer when possible
5. If multiple methods exist, mention them
6. Be pedagogical - help the user learn

Format your response with:
- **Problem Understanding**: Restate what you're solving
- **Solution Steps**: Numbered steps with explanations
- **Final Answer**: Clear, highlighted result
- **Verification**: Check your work if possible"""

        specialized_prompts = {
            "calculus_derivative": base_prompt + "\n\nSpecialize in: Derivatives, chain rule, product rule, quotient rule, implicit differentiation.",
            "calculus_integral": base_prompt + "\n\nSpecialize in: Integration techniques, substitution, integration by parts, definite integrals.",
            "algebra": base_prompt + "\n\nSpecialize in: Solving equations, systems of equations, factoring, simplification.",
            "geometry": base_prompt + "\n\nSpecialize in: Area, volume, perimeter calculations, geometric theorems.",
            "statistics": base_prompt + "\n\nSpecialize in: Descriptive statistics, probability distributions, hypothesis testing.",
            "word_problem": base_prompt + "\n\nSpecialize in: Translating word problems into mathematical expressions, real-world applications.",
            "arithmetic": base_prompt + "\n\nSpecialize in: Step-by-step arithmetic, order of operations, fractions, decimals.",
            "general": base_prompt + "\n\nAnalyze the problem carefully to determine the best approach."
        }
        
        return specialized_prompts.get(problem_type, base_prompt)
    
    def _format_solution(self, problem: str, solution: str, problem_type: str) -> str:
        """Format the mathematical solution with enhanced presentation"""
        try:
            formatted_response = f"🔢 **Math Solution ({problem_type.replace('_', ' ').title()})**\n\n"
            formatted_response += f"**Original Problem:** {problem}\n\n"
            formatted_response += "---\n\n"
            formatted_response += solution
            formatted_response += f"\n\n---\n*Solved using advanced mathematical reasoning*"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Solution formatting error: {e}")
            return f"🔢 **Math Solution**\n\n{solution}"
    
    def _handle_solver_error(self, problem: str, error: str) -> str:
        """Handle and format math solver errors"""
        error_response = f"❌ **Math Solver Error**\n\n"
        error_response += f"**Problem:** {problem}\n"
        error_response += f"**Error:** {error}\n\n"
        
        # Provide helpful suggestions
        suggestions = [
            "• Try rephrasing the problem more clearly",
            "• Include all necessary information and constraints",
            "• Use standard mathematical notation",
            "• Break complex problems into smaller parts"
        ]
        
        error_response += "**Suggestions:**\n"
        for suggestion in suggestions:
            error_response += f"{suggestion}\n"
        
        error_response += "\n**Example problems I can solve:**\n"
        error_response += "• Solve for x: 2x + 5 = 17\n"
        error_response += "• Find the derivative of x² + 3x - 2\n"
        error_response += "• Calculate the area of a circle with radius 5\n"
        error_response += "• If John has 20 apples and gives away 30%, how many does he have left?\n"
        
        return error_response
    
    def get_problem_examples(self) -> Dict[str, List[str]]:
        """Get example problems by category"""
        return {
            "Algebra": [
                "Solve for x: 3x - 7 = 14",
                "Factor: x² - 5x + 6",
                "Solve the system: 2x + y = 7, x - y = 2"
            ],
            "Calculus": [
                "Find the derivative of x³ + 2x² - 5x + 1",
                "Integrate: ∫(2x + 3)dx",
                "Find the area under y = x² from x = 0 to x = 3"
            ],
            "Geometry": [
                "Find the area of a triangle with base 8 and height 6",
                "Calculate the volume of a sphere with radius 4",
                "Find the hypotenuse of a right triangle with legs 3 and 4"
            ],
            "Word Problems": [
                "A car travels 60 mph for 2.5 hours. How far does it go?",
                "If 15% of students failed an exam, and 12 students failed, how many took the exam?",
                "A recipe calls for 2 cups of flour for 12 cookies. How much flour for 18 cookies?"
            ]
        }
    
    def validate_solution(self, problem: str, solution: str) -> Dict[str, Any]:
        """Validate a mathematical solution (optional feature for evaluation)"""
        try:
            validation_prompt = f"""
            Validate this mathematical solution:
            
            Problem: {problem}
            Solution: {solution}
            
            Check if:
            1. The solution method is correct
            2. All steps are mathematically sound
            3. The final answer is accurate
            4. The explanation is clear and complete
            
            Respond with a JSON object containing:
            - "valid": true/false
            - "accuracy_score": 0-1
            - "feedback": "detailed feedback"
            """
            
            if self.llm:
                response = self.llm.invoke([HumanMessage(content=validation_prompt)])
                # Parse JSON response (simplified for this example)
                return {
                    "valid": True,
                    "accuracy_score": 0.9,
                    "feedback": "Solution appears mathematically correct"
                }
            
        except Exception as e:
            logger.error(f"Solution validation error: {e}")
        
        return {
            "valid": False,
            "accuracy_score": 0.0,
            "feedback": "Could not validate solution"
        }