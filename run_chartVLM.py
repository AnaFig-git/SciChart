import os
import json
import logging
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_dir)

chartvlm_main_dir = os.path.join(current_dir, "ChartVLM")
sys.path.append(chartvlm_main_dir)

from tools.ChartVLM import infer_ChartVLM
from instructions import instruction

# Replace with your own cache directory
MODEL_PATH = ""

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def get_qa_type(data: dict) -> str:
    for key in ['CH', 'EN']:
        if any(key in k for k in data.keys()):
            return key
    return 'EN'

if __name__ == '__main__':
    fileName = input("Please enter the dataset file name to run (e.g., ReaQA-EN-val): ").replace(".json", "")
    data_file = Path(f"data/{fileName}.json")
    output_file = Path(f"output/{fileName}_ChartVLMResult.jsonl")

    with data_file.open(encoding='utf-8') as f:
        dataDict = json.load(f)

    QAType = get_qa_type(dataDict)
    results = []
    total = len(dataDict)

    print(f"Starting inference: {fileName} (Total: {total} items)")

    for i, (key, value) in enumerate(dataDict.items()):
        imgPrefix = 'Basic' if 'question_type' in value else 'Rea'
        img_path = f"data/img/{imgPrefix}/{value['image_id']}.jpg"
        
        if not Path(img_path).exists():
            logging.warning(f"Image not found: {img_path}")
            continue

        prefix = instruction.get(
            f"{value.get('question_type', 'Rea')}_{QAType}",
            instruction[f'Rea_{QAType}']
        )

        text_prompt = f"{prefix}\nQuestion: {value['question']}"

        response = infer_ChartVLM(img_path, text_prompt, MODEL_PATH)

        clean_response = str(response).strip()

        print(f"[{i+1}/{total}] ID: {value['image_id']} | Ans: {clean_response.replace(chr(10), ' ').replace('</s>', '')}")

        result = {
            'image_id': value['image_id'],
            'question': value['question'],
            'ans_gt': value['answer'],
            'ans_pre': clean_response
        }
        results.append(result)

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with output_file.open('w', encoding='utf-8') as rf:
        for result in results:
            rf.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Done! Results saved to: {output_file}")
