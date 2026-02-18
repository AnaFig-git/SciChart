import os
import json
import torch
import logging
from pathlib import Path
from PIL import Image

# Set Hugging Face mirror for fast downloading on AutoDL
from huggingface_hub import snapshot_download

from transformers import AutoModelForCausalLM, AutoTokenizer
from instructions import instruction

# Replace with your own cache directory
CACHE_DIR = ""
REPO_ID = "Qwen/Qwen-VL"  # Hugging Face repo ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def get_model_path():
    print(f"Checking model path, download directory: {CACHE_DIR} ...")
    model_dir = snapshot_download(
        repo_id=REPO_ID, 
        cache_dir=CACHE_DIR,
        resume_download=True
    )
    print(f"Model ready, path: {model_dir}")
    return model_dir

def load_model(model_path):
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    dtype_args = {"bf16": True} if torch.cuda.is_bf16_supported() else {"fp16": True}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", 
        trust_remote_code=True, 
        **dtype_args
    ).eval()
    return model, tokenizer

def get_qa_type(data: dict) -> str:
    for key in ['CH', 'EN']:
        if any(key in k for k in data.keys()):
            return key
    return 'EN'

if __name__ == '__main__':
    model_path = get_model_path()
    model, tokenizer = load_model(model_path)

    fileName = input("Please enter the dataset file name to run (e.g., BasicQA-EN-Test): ").replace(".json", "")
    data_file = Path(f"data/{fileName}.json")
    output_file = Path(f"output/{fileName}_QwenBaseResult.jsonl")

    with data_file.open(encoding='utf-8') as f:
        dataDict = json.load(f)

    QAType = get_qa_type(dataDict)
    results = []
    total = len(dataDict)
    
    print(f"Starting inference: {fileName} (Total {total} items)")

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
        
        query = tokenizer.from_list_format([
            {'image': img_path},
            {'text': f"{prefix}\nQuestion: {value['question']}\nAnswer:"},
        ])
        
        inputs = tokenizer(query, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            pred = model.generate(**inputs, max_new_tokens=128)
        
        full_response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        
        if "Answer:" in full_response:
            response = full_response.split("Answer:")[-1].strip()
        else:
            response = full_response.replace(query, "").strip()
        
        print(f"[{i+1}/{total}] ID: {value['image_id']} | Ans: {response.replace(chr(10), ' ')}")

        result = {
            'image_id': value['image_id'],
            'question': value['question'],
            'ans_gt': value['answer'],
            'ans_pre': response
        }
        results.append(result)

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with output_file.open('w', encoding='utf-8') as rf:
        for result in results:
            rf.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Done! Results saved to: {output_file}")
