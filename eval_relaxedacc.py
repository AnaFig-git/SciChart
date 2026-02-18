import json
import logging
from pathlib import Path
from typing import List, Dict, Union

"""
This module selects evaluation files from the output directory and systematically assesses the correctness of generated answers by comparing them against ground truth references.
The evaluation results are used for following analysis and reporting.
"""

# Basic Logging Configuration
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def real_correctness(p: str, t: str) -> bool:
    truth_pairs = {
        "yes": {"是", "yes"},
        "no": {"否", "no"},
        "是": {"是", "yes"},
        "否": {"否", "no"}
    }
    return truth_pairs.get(p.lower(), set()).intersection({t.lower()}) != set()

def relaxed_correctness(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05
) -> bool:
    
    def _to_float(text: str) -> Union[float, None]:
        try:
            text = text.strip().lower()
            if text.endswith('%'):
                return float(text.rstrip('%')) / 100.0
            return float(text)
        except (ValueError, TypeError):
            return None

    p_float = _to_float(prediction)
    t_float = _to_float(target)
    
    if p_float is not None and t_float is not None:
        try:
            relative_change = abs(p_float - t_float) / abs(t_float)
            return relative_change <= max_relative_change
        except ZeroDivisionError:
            logger.warning("Division by zero error detected")
            return p_float == t_float
    return real_correctness(prediction, target)

def evaluate_relaxed_accuracy(entries: List[str]) -> float:
    scores = []
    
    for line in entries:
        try:
            elem = json.loads(line.strip().replace("'", "\""))
            
            gt_answers = elem['ans_gt']
            if isinstance(gt_answers, str):
                gt_answers = [gt_answers]
                
            score = max(
                relaxed_correctness(elem['ans_pre'].strip(), ann)
                for ann in gt_answers
            )
            
            logger.info(
                f"ground truth references：{gt_answers}，generated answer：{elem['ans_pre']}，score：{score}"
            )
            scores.append(score)
            
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Invalid data format: {str(e)}")
            continue
            
    return sum(scores) / len(scores) if scores else 0.0

if __name__ == "__main__":
    try:
        fileName = input("Enter the filename you want to evaluate:").replace(".jsonl", "")
        
        result_path = Path(f"output/{fileName}.jsonl")
        if not result_path.exists():
            raise FileNotFoundError(f"File does not exist: {result_path}")
            
        with result_path.open(encoding="utf-8") as f:
            contents = f.readlines()
            
        accuracy = evaluate_relaxed_accuracy(contents)
        print({"relaxed_accuracy": round(accuracy, 4)})  # Rounded to 4 Decimal Places
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise
