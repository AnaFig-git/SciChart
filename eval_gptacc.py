import json
import logging
from pathlib import Path
from gpt_acc import eval_gpt_acc  

"""
This module selects evaluation files from the output directory, then systematically evaluates the generated answers against ground truth references and questions.
It leverages the GPT API to perform comprehensive accuracy assessments through multi-factor analysis.
The evaluation results are saved in a structured JSON format for subsequent analysis.
"""

# Basic Logging Configuration
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def safe_parse_line(line: str) -> dict:
    try:
        return json.loads(line.replace("'", "\""))  
    except json.JSONDecodeError:
        logging.error("Data parsing failed: %s", line)
        return None

if __name__ == "__main__":
    try:
       
        fileName = input("Enter the filename you want to evaluate: ").replace(".josnl", "")
        
        
        result_path = Path(f"output/{fileName}.jsonl")
        if not result_path.exists():
            raise FileNotFoundError(f"File does not exist: {result_path}")
            
        
        qa_score_set = []
        
        with result_path.open(encoding="utf-8") as f:
            for line in f:
                content = safe_parse_line(line.strip())
                if not content:
                    continue
                
                try:
                    qa_score = eval_gpt_acc(
                        content['question'],
                        content['ans_gt'],
                        content['ans_pre'],
                        ""
                    )
                    qa_score_set.append(qa_score)
                    #print(qa_score)  
                except KeyError as e:
                    logging.warning(f"Missing data field: {e}")
                    
        if qa_score_set:
            qa_score_type = sum(qa_score_set) / len(qa_score_set)
            result = {'gpt_acc': round(qa_score_type, 4)}  
            print(result) 

                
            
    except Exception as e:
        logging.error(f"Execution failed: {str(e)}")
        raise
