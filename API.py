import os
import json
import time
import base64
import logging
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from PIL import Image
from instructions import instruction

"""
This module selects a sub-dataset from the data directory and retrieves corresponding reference instructions from "instruction.py" based on the question category.
It then selects a model, invokes the corresponding API to solve chart question answering tasks, and saves the generated results in the output directory for subsequent evaluation.
"""

# Base Configuration
API_SECRET_KEY = os.getenv("API_KEY", "sk-default-key")
MAX_RETRY = 3

# Basic Logging Configuration
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def encode_image_to_base64(img_path: str) -> str:
    try:
        with Image.open(img_path) as img:
            buffer = BytesIO()
            img.save(buffer, format=img.format)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Image Processing Failed: {img_path} - {str(e)}")
        raise


def simple_multimodal_conversation_call(img, question, prefix):
    client = OpenAI(api_key=API_SECRET_KEY)

    for _ in range(MAX_RETRY):
        try:
            base64_image = encode_image_to_base64(img)
            messages = [
                {"role": "system", "content": prefix},
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ]}
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",  # Target API Model
                messages=messages,
                timeout=15
            )

            if response.choices:
                return response.choices[0].message.content

        except Exception as e:
            logging.warning(f"API Call Failed: {str(e)}")
            time.sleep(10)

    logging.error(f"Maximum Retry Attempts Exceeded: {img}")
    return None


def get_qa_type(data: dict) -> str:
    for key in ['CH', 'EN']:
        if any(key in k for k in data.keys()):
            return key
    return 'EN'


if __name__ == '__main__':

    fileName = input("Enter the dataset to run:").replace(".json", "")

    data_file = Path(f"data/{fileName}.json")
    output_file = Path(f"output/{fileName}Result.jsonl")

    try:
        with data_file.open(encoding='utf-8') as f:
            dataDict = json.load(f)

        QAType = get_qa_type(dataDict)
        results = []
        for key, value in dataDict.items():
            imgPrefix = ''
            if 'question_type' in value:
                imgPrefix = 'Basic'
            else:
                imgPrefix = 'Rea'
            img_path = f"data/img/{imgPrefix}/{value['image_id']}.jpg"
            if not Path(img_path).exists():
                logging.warning(f"Image Not Found: {img_path}")
                continue

            prefix = instruction.get(
                f"{value.get('question_type', 'Rea')}_{QAType}",
                instruction[f'Rea_{QAType}']
            )

            ans_pre = simple_multimodal_conversation_call(
                img_path,
                value['question'],
                prefix
            )

            result = {
                'image_id': value['image_id'],
                'question': value['question'],
                'ans_gt': value['answer'],
                'ans_pre': ans_pre
            }
            results.append(result)
            # print(result)

        output_file.parent.mkdir(exist_ok=True)
        with output_file.open('w', encoding='utf-8') as rf:
            for result in results:
                rt = json.dumps(result)
                rf.write(rt + '\n')


    except Exception as e:
        logging.error(f"Execution Failed: {str(e)}")
        raise
