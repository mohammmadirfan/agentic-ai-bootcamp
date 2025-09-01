import json
import logging
import random
import re
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Sample GSM8K (Grade School Math) problems
GSM8K_PROBLEMS = [
    {
        "problem": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "18",
        "category": "word_problem"
    },
    {
        "problem": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "3",
        "category": "basic_arithmetic"
    },
    {
        "problem": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": "65000",
        "category": "percentages"
    },
    {
        "problem": "There are 15 trees in the grove. Grove workers will plant trees today. After they are done there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "6",
        "category": "subtraction"
    },
    {
        "problem": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "5",
        "category": "addition"
    },
    {
        "problem": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "39",
        "category": "multi_step"
    },
    {
        "problem": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "8",
        "category": "subtraction"
    },
    {
        "problem": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "9",
        "category": "addition"
    },
    {
        "problem": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "29",
        "category": "multi_step"
    },
    {
        "problem": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "33",
        "category": "multi_step"
    }
]

def run_gsm8k_evaluation(agent_controller, num_problems: int = 10) -> Dict[str, Any]:
    """
    Run GSM8K evaluation to test mathematical reasoning
    
    Args:
        agent_controller: The agent controller instance
        num_problems: Number of problems to test
        
    Returns:
        Evaluation results dictionary
    """
    try:
        # Select random problems for evaluation
        test_problems = random.sample(GSM8K_PROBLEMS, min(num_problems, len(GSM8K_PROBLEMS)))
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_problems": len(test_problems),
            "correct_answers": 0,
            "accuracy": 0.0,
            "details": [],
            "category_performance": {}
        }
        
        logger.info(f"Starting GSM8K evaluation with {len(test_problems)} problems")
        
        for i, item in enumerate(test_problems, 1):
            problem = item["problem"]
            expected_answer = str(item["answer"])
            category = item["category"]
            
            try:
                # Get agent response
                response = agent_controller.process_query(problem)
                agent_response = response["response"]
                
                # Extract numerical answer from response
                extracted_answer = _extract_numerical_answer(agent_response)
                is_correct = extracted_answer == expected_answer
                
                if is_correct:
                    results["correct_answers"] += 1
                
                # Track category performance
                if category not in results["category_performance"]:
                    results["category_performance"][category] = {"correct": 0, "total": 0}
                
                results["category_performance"][category]["total"] += 1
                if is_correct:
                    results["category_performance"][category]["correct"] += 1
                
                # Store detailed result
                results["details"].append({
                    "problem": problem,
                    "expected_answer": expected_answer,
                    "extracted_answer": extracted_answer,
                    "full_response": agent_response[:300] + "..." if len(agent_response) > 300 else agent_response,
                    "tool_used": response["tool_used"],
                    "correct": is_correct,
                    "category": category
                })
                
                logger.info(f"GSM8K problem {i}/{len(test_problems)}: {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                logger.error(f"Error processing GSM8K problem {i}: {e}")
                results["details"].append({
                    "problem": problem,
                    "expected_answer": expected_answer,
                    "extracted_answer": None,
                    "full_response": f"Error: {str(e)}",
                    "tool_used": "Error",
                    "correct": False,
                    "category": category,
                    "error": str(e)
                })
        
        # Calculate final accuracy
        results["accuracy"] = results["correct_answers"] / results["total_problems"]
        
        # Calculate category accuracies
        for category, perf in results["category_performance"].items():
            perf["accuracy"] = perf["correct"] / perf["total"] if perf["total"] > 0 else 0
        
        # Save results
        _save_evaluation_results("gsm8k", results)
        
        logger.info(f"GSM8K evaluation completed: {results['accuracy']:.2%} accuracy")
        return results
        
    except Exception as e:
        logger.error(f"GSM8K evaluation error: {e}")
        return {
            "error": str(e),
            "accuracy": 0.0,
            "total_problems": 0,
            "correct_answers": 0
        }

def _extract_numerical_answer(response: str) -> str:
    """Extract numerical answer from agent response"""
    # Look for numbers in the response
    # Try to find the final answer or conclusion
    lines = response.split('\n')
    
    # Look for explicit answer patterns
    answer_patterns = [
        r'answer(?:\s+is)?:?\s*(\d+(?:\.\d+)?)',
        r'result(?:\s+is)?:?\s*(\d+(?:\.\d+)?)',
        r'solution(?:\s+is)?:?\s*(\d+(?:\.\d+)?)',
        r'final(?:\s+answer)?:?\s*(\d+(?:\.\d+)?)',
        r'therefore:?\s*(\d+(?:\.\d+)?)',
        r'equals?\s*(\d+(?:\.\d+)?)',
        r'=\s*(\d+(?:\.\d+)?)',
        r'\$(\d+(?:\.\d+)?)',  # Dollar amounts
        r'(\d+(?:\.\d+)?)\s*dollars?',
        r'(\d+(?:\.\d+)?)\s*(?:items?|things?|pieces?|units?)'
    ]
    
    # Try each pattern
    for pattern in answer_patterns:
        matches = re.findall(pattern, response.lower())
        if matches:
            # Return the last match (usually the final answer)
            return str(int(float(matches[-1])))
    
    # Fallback: find all numbers and return the last one
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    if numbers:
        return str(int(float(numbers[-1])))
    
    return "No answer found"

def _save_evaluation_results(eval_type: str, results: Dict[str, Any]):
    """Save evaluation results to file"""
    try:
        results_dir = Path("data/results/answers")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{eval_type}_evaluation_{timestamp}.json"
        
        with open(results_dir / filename, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")

def compare_evaluation_results(eval_type: str) -> Dict[str, Any]:
    """Compare current evaluation with historical results"""
    try:
        results_dir = Path("data/results/answers")
        if not results_dir.exists():
            return {"error": "No evaluation history found"}
        
        # Find all evaluation files of this type
        pattern = f"{eval_type}_evaluation_*.json"
        eval_files = list(results_dir.glob(pattern))
        
        if len(eval_files) < 2:
            return {"error": "Need at least 2 evaluations to compare"}
        
        # Sort by timestamp (newest first)
        eval_files.sort(key=lambda x: x.name, reverse=True)
        
        # Load latest two evaluations
        with open(eval_files[0]) as f:
            latest = json.load(f)
        with open(eval_files[1]) as f:
            previous = json.load(f)
        
        comparison = {
            "latest_accuracy": latest.get("accuracy", 0),
            "previous_accuracy": previous.get("accuracy", 0),
            "improvement": latest.get("accuracy", 0) - previous.get("accuracy", 0),
            "latest_date": latest.get("timestamp", ""),
            "previous_date": previous.get("timestamp", ""),
            "trend": "improving" if latest.get("accuracy", 0) > previous.get("accuracy", 0) else "declining"
        }
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing evaluation results: {e}")
        return {"error": str(e)}

def get_evaluation_history(eval_type: str) -> List[Dict[str, Any]]:
    """Get historical evaluation results"""
    try:
        results_dir = Path("data/results/answers")
        if not results_dir.exists():
            return []
        
        pattern = f"{eval_type}_evaluation_*.json"
        eval_files = list(results_dir.glob(pattern))
        
        history = []
        for file_path in sorted(eval_files, key=lambda x: x.name):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    history.append({
                        "timestamp": data.get("timestamp", ""),
                        "accuracy": data.get("accuracy", 0),
                        "total_questions": data.get("total_questions", 0),
                        "correct_answers": data.get("correct_answers", 0),
                        "filename": file_path.name
                    })
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return history
        
    except Exception as e:
        logger.error(f"Error getting evaluation history: {e}")
        return []

def generate_evaluation_report(eval_type: str) -> str:
    """Generate a comprehensive evaluation report"""
    try:
        results_dir = Path("data/results/answers")
        pattern = f"{eval_type}_evaluation_*.json"
        eval_files = list(results_dir.glob(pattern))
        
        if not eval_files:
            return f"No {eval_type} evaluation results found."
        
        # Load latest evaluation
        latest_file = max(eval_files, key=lambda x: x.name)
        with open(latest_file) as f:
            latest_results = json.load(f)
        
        # Generate report
        report = f"# {eval_type.upper()} Evaluation Report\n\n"
        report += f"**Evaluation Date:** {latest_results.get('timestamp', 'Unknown')}\n"
        report += f"**Overall Accuracy:** {latest_results.get('accuracy', 0):.2%}\n"
        report += f"**Questions Answered:** {latest_results.get('correct_answers', 0)}/{latest_results.get('total_questions', 0)}\n\n"
        
        # Category breakdown
        if "category_performance" in latest_results:
            report += "## Performance by Category\n\n"
            for category, perf in latest_results["category_performance"].items():
                accuracy = perf.get("accuracy", 0)
                correct = perf.get("correct", 0)
                total = perf.get("total", 0)
                report += f"- **{category.title()}:** {accuracy:.2%} ({correct}/{total})\n"
        
        # Recent trends (if multiple evaluations exist)
        comparison = compare_evaluation_results(eval_type)
        if "improvement" in comparison:
            report += f"\n## Recent Trends\n\n"
            trend = "📈 Improving" if comparison["improvement"] > 0 else "📉 Declining"
            report += f"**Trend:** {trend} ({comparison['improvement']:+.2%})\n"
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating evaluation report: {e}")
        return f"Error generating {eval_type} evaluation report: {str(e)}"