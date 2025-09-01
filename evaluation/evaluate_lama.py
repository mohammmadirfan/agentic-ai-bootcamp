import json
import logging
import random
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Sample LAMA (Language Model Analysis) test questions
LAMA_QUESTIONS = [
    {
        "question": "The capital of France is",
        "answer": "Paris",
        "category": "geography"
    },
    {
        "question": "Water boils at __ degrees Celsius",
        "answer": "100",
        "category": "science"
    },
    {
        "question": "The largest planet in our solar system is",
        "answer": "Jupiter",
        "category": "astronomy"
    },
    {
        "question": "Shakespeare wrote",
        "answer": "Hamlet",
        "category": "literature"
    },
    {
        "question": "The formula for the area of a circle is",
        "answer": "πr²",
        "category": "mathematics"
    },
    {
        "question": "The first president of the United States was",
        "answer": "George Washington",
        "category": "history"
    },
    {
        "question": "DNA stands for",
        "answer": "Deoxyribonucleic acid",
        "category": "biology"
    },
    {
        "question": "The speed of light is approximately __ meters per second",
        "answer": "300000000",
        "category": "physics"
    },
    {
        "question": "The currency of Japan is",
        "answer": "Yen",
        "category": "geography"
    },
    {
        "question": "The smallest unit of matter is",
        "answer": "atom",
        "category": "chemistry"
    }
]

def run_lama_evaluation(agent_controller, num_questions: int = 10) -> Dict[str, Any]:
    """
    Run LAMA evaluation to test factual knowledge
    
    Args:
        agent_controller: The agent controller instance
        num_questions: Number of questions to test
        
    Returns:
        Evaluation results dictionary
    """
    try:
        # Select random questions for evaluation
        test_questions = random.sample(LAMA_QUESTIONS, min(num_questions, len(LAMA_QUESTIONS)))
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(test_questions),
            "correct_answers": 0,
            "accuracy": 0.0,
            "details": [],
            "category_performance": {}
        }
        
        logger.info(f"Starting LAMA evaluation with {len(test_questions)} questions")
        
        for i, item in enumerate(test_questions, 1):
            question = item["question"]
            expected_answer = item["answer"].lower()
            category = item["category"]
            
            try:
                # Get agent response
                response = agent_controller.process_query(question)
                agent_answer = response["response"].lower()
                
                # Check if answer is correct (simple string matching)
                is_correct = expected_answer in agent_answer or any(
                    word in agent_answer for word in expected_answer.split()
                    if len(word) > 2
                )
                
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
                    "question": question,
                    "expected_answer": item["answer"],
                    "agent_answer": response["response"][:200] + "..." if len(response["response"]) > 200 else response["response"],
                    "tool_used": response["tool_used"],
                    "correct": is_correct,
                    "category": category
                })
                
                logger.info(f"LAMA question {i}/{len(test_questions)}: {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                logger.error(f"Error processing LAMA question {i}: {e}")
                results["details"].append({
                    "question": question,
                    "expected_answer": item["answer"],
                    "agent_answer": f"Error: {str(e)}",
                    "tool_used": "Error",
                    "correct": False,
                    "category": category,
                    "error": str(e)
                })
        
        # Calculate final accuracy
        results["accuracy"] = results["correct_answers"] / results["total_questions"]
        
        # Calculate category accuracies
        for category, perf in results["category_performance"].items():
            perf["accuracy"] = perf["correct"] / perf["total"] if perf["total"] > 0 else 0
        
        # Save results
        _save_evaluation_results("lama", results)
        
        logger.info(f"LAMA evaluation completed: {results['accuracy']:.2%} accuracy")
        return results
        
    except Exception as e:
        logger.error(f"LAMA evaluation error: {e}")
        return {
            "error": str(e),
            "accuracy": 0.0,
            "total_questions": 0,
            "correct_answers": 0
        }

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