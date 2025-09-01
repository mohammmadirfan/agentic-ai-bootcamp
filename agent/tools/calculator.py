import re
import math
import logging
from typing import Any, Dict
from typing import List
import ast
import operator

logger = logging.getLogger(__name__)

class CalculatorTool:
    """Safe calculator tool for basic mathematical operations"""
    
    def __init__(self):
        """Initialize the calculator tool"""
        self.description = "Perform basic mathematical calculations and arithmetic operations"
        self.capabilities = [
            "Basic arithmetic (+, -, *, /)",
            "Exponents and roots",
            "Trigonometric functions",
            "Logarithms",
            "Percentages",
            "Unit conversions"
        ]
        
        # Safe operations mapping
        self.safe_operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        # Safe functions
        self.safe_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'exp': math.exp,
            'sqrt': math.sqrt,
            'ceil': math.ceil,
            'floor': math.floor,
            'degrees': math.degrees,
            'radians': math.radians,
            'factorial': math.factorial,
            'pi': math.pi,
            'e': math.e
        }
    
    def calculate(self, expression: str) -> str:
        """
        Safely evaluate mathematical expressions
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Formatted calculation result
        """
        try:
            # Clean and preprocess the expression
            cleaned_expr = self._preprocess_expression(expression)
            
            # Try to evaluate using safe evaluation
            result = self._safe_eval(cleaned_expr)
            
            # Format the result
            return self._format_result(expression, result)
            
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return self._handle_calculation_error(expression, str(e))
    
    def _preprocess_expression(self, expr: str) -> str:
        """Clean and preprocess mathematical expression"""
        # Remove common words and convert to math expression
        expr = expr.lower().strip()
        
        # Handle percentage calculations
        if "%" in expr:
            expr = self._handle_percentages(expr)
        
        # Handle word-based operations
        replacements = {
            " plus ": " + ",
            " minus ": " - ",
            " times ": " * ",
            " multiplied by ": " * ",
            " divided by ": " / ",
            " to the power of ": " ** ",
            " squared": " ** 2",
            " cubed": " ** 3",
            "square root of ": "sqrt(",
            "sqrt of ": "sqrt(",
            "sin of ": "sin(",
            "cos of ": "cos(",
            "tan of ": "tan(",
            "log of ": "log(",
            "ln of ": "log(",
        }
        
        for word, symbol in replacements.items():
            expr = expr.replace(word, symbol)
        
        # Handle implicit multiplication (e.g., "2(3+4)" -> "2*(3+4)")
        expr = re.sub(r'(\d)\(', r'\1*(', expr)
        expr = re.sub(r'\)(\d)', r')*\1', expr)
        
        # Clean up extra spaces
        expr = re.sub(r'\s+', '', expr)
        
        return expr
    
    def _handle_percentages(self, expr: str) -> str:
        """Handle percentage calculations"""
        # Pattern: "X% of Y" -> "X * Y / 100"
        percent_pattern = r'(\d+(?:\.\d+)?)%\s*of\s*(\d+(?:\.\d+)?)'
        expr = re.sub(percent_pattern, r'(\1 * \2 / 100)', expr)
        
        # Pattern: "increase X by Y%" -> "X * (1 + Y/100)"
        increase_pattern = r'increase\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)%'
        expr = re.sub(increase_pattern, r'(\1 * (1 + \2/100))', expr)
        
        # Pattern: "decrease X by Y%" -> "X * (1 - Y/100)"
        decrease_pattern = r'decrease\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)%'
        expr = re.sub(decrease_pattern, r'(\1 * (1 - \2/100))', expr)
        
        # Simple percentage conversion: "25%" -> "25/100"
        expr = re.sub(r'(\d+(?:\.\d+)?)%', r'(\1/100)', expr)
        
        return expr
    
    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate mathematical expression using AST"""
        try:
            # Parse the expression
            node = ast.parse(expression, mode='eval')
            
            # Evaluate the AST
            return self._eval_node(node.body)
            
        except SyntaxError:
            # Fallback to function-based evaluation
            return self._eval_with_functions(expression)
    
    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.safe_operations.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.safe_operations.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in self.safe_functions:
                args = [self._eval_node(arg) for arg in node.args]
                return self.safe_functions[func_name](*args)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id in self.safe_functions:
                return self.safe_functions[node.id]
            else:
                raise ValueError(f"Unsupported variable: {node.id}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    
    def _eval_with_functions(self, expression: str) -> float:
        """Evaluate expressions with function calls"""
        # Create a safe namespace with only allowed functions
        safe_dict = {
            "__builtins__": {},
            **self.safe_functions
        }
        
        try:
            result = eval(expression, safe_dict)
            return float(result)
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {str(e)}")
    
    def _format_result(self, original_expr: str, result: float) -> str:
        """Format the calculation result"""
        try:
            # Format number appropriately
            if result == int(result):
                formatted_result = str(int(result))
            elif abs(result) < 0.001 or abs(result) > 1e6:
                formatted_result = f"{result:.2e}"
            else:
                formatted_result = f"{result:.6f}".rstrip('0').rstrip('.')
            
            response = f"🧮 **Calculation Result**\n\n"
            response += f"**Expression:** `{original_expr}`\n"
            response += f"**Result:** `{formatted_result}`\n\n"
            
            # Add additional context for special numbers
            if abs(result - math.pi) < 0.0001:
                response += "📝 *This is approximately π (pi)*\n"
            elif abs(result - math.e) < 0.0001:
                response += "📝 *This is approximately e (Euler's number)*\n"
            elif result == 0:
                response += "📝 *Result is zero*\n"
            elif result < 0:
                response += "📝 *Result is negative*\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Result formatting error: {e}")
            return f"Calculation completed but formatting failed: {result}"
    
    def _handle_calculation_error(self, expression: str, error: str) -> str:
        """Handle and format calculation errors"""
        error_response = f"❌ **Calculation Error**\n\n"
        error_response += f"**Expression:** `{expression}`\n"
        error_response += f"**Error:** {error}\n\n"
        
        # Provide helpful suggestions
        suggestions = []
        
        if "unsupported" in error.lower():
            suggestions.append("• Try using basic operations: +, -, *, /, **")
            suggestions.append("• Available functions: sin, cos, tan, log, sqrt, abs, round")
        
        if "syntax" in error.lower():
            suggestions.append("• Check for balanced parentheses")
            suggestions.append("• Use * for multiplication (e.g., 2*3, not 2×3)")
        
        if "division by zero" in error.lower():
            suggestions.append("• Cannot divide by zero")
            suggestions.append("• Check your expression for zero denominators")
        
        if suggestions:
            error_response += "**Suggestions:**\n"
            for suggestion in suggestions:
                error_response += f"{suggestion}\n"
        
        error_response += "\n**Examples of valid expressions:**\n"
        error_response += "• `2 + 3 * 4`\n"
        error_response += "• `sqrt(16) + 5`\n"
        error_response += "• `sin(pi/2)`\n"
        error_response += "• `10% of 150`\n"
        
        return error_response
    
    def test_calculation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test the calculator with predefined test cases"""
        results = {
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for test in test_cases:
            expr = test["expression"]
            expected = test["expected"]
            tolerance = test.get("tolerance", 1e-6)
            
            try:
                result = self._safe_eval(self._preprocess_expression(expr))
                
                if abs(result - expected) <= tolerance:
                    results["passed"] += 1
                    results["details"].append({
                        "expression": expr,
                        "expected": expected,
                        "actual": result,
                        "status": "PASS"
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "expression": expr,
                        "expected": expected,
                        "actual": result,
                        "status": "FAIL"
                    })
                    
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "expression": expr,
                    "expected": expected,
                    "actual": None,
                    "error": str(e),
                    "status": "ERROR"
                })
        
        results["accuracy"] = results["passed"] / (results["passed"] + results["failed"]) if (results["passed"] + results["failed"]) > 0 else 0
        
        return results