# 📊 SciChart
This repository contains the code to evaluate models on SciChart from the paper [SciChart: Visual Question Answering and Reasoning for
 Scientific Spectral Chart].

*🤗 This codebase is released as Version v1.0. We are dedicated to its continuous improvement. If you have any questions or suggestions, you are welcome to open an issue or submit a pull request for new features or bug fixes.*

## 👋 Introduction

Charts play a pivotal role in scientific research, offering a concise and visual way to present complex data. For Multimodal Large Language Models (MLLMs), the ability to comprehend charts is critical, as it requires both visual perception and reasoning that bridges graphical and textual information. However, existing Chart QA datasets are monolingual with simple questions, making current evaluation benchmarks inadequate for the rapid advancements in MLLM performance. Therefore, we propose a multilingual scientific spectral Chart QA dataset, termed SciChart. We design two tasks, basic question answering (BasicQA) and reasoning-based question answering (ReaQA), to evaluate the models' ability to 1) directly extract information from charts, and 2) understand the textual and visual information for reasoning. We build 1,100 ReaQA and over 10,000 BasicQA samples. All samples are manually curated and annotated by human experts. We also conduct extensive experiments with state-of-the-art models to establish SciChart benchmarks. Experimental results show a huge gap between the performance of existing models (Claude-3.7 45.12%) and humans (83.84%).

<div align=center>
<img src="example_image/charts.png">
</div>

<div align=center>
<img src="example_image/examples.png" >
</div> 

## 📉 Dataset Statistics 
SciChart covers **6 mainstream spectral chart types**. The dataset is split into **BasicQA** and **ReaQA** tasks, supporting both Chinese and English.

| Type | Language | Train | Val | Test | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BasicQA** | CH & EN | 6,000 | 2,010 | 2,009 | **10,019** |
| **ReaQA** | CH & EN | - | 550 | 550 | **1,100** |

### 🎯 Task Definition
We design two main tasks to comprehensively evaluate MLLMs:

**BasicQA (Information Extraction):** Focuses on perceiving explicit information.   
 
 *Subtasks:* Peak Number, Peak Position, Peak Value, FWHM (Full Width at Half Maximum), and Shape.   
  
  *Note:* The FWHM task involves complex multi-step calculation based on visual coordinates. 

**ReaQA (Reasoning):** Focuses on implicit information and scientific logic.   
  
  *Subtasks:* Includes **Abductive**, **Deductive**, and **Inductive** reasoning questions presented as Multi-choice or True/False.  

## Data Structure
 Responses are generated using queries as input, which contain  the charts and questions that SciChart uses to evaluate models. The structure is as follows:
```
{
    "1": {
        "image_id": ...<str>,
        "question_type": ...<str>,
        "question": ...<str>,
        "answer": ...<str>
    },
    "2": {
        "image_id": ...<str>,
        "question_type": ...<str>,
        "question": ...<str>,
        "answer": ...<str>
    }
    ...
}
```
## Image Data
Please download image data to **data/** from link: https://drive.google.com/file/d/1M3kzYqLK26KLHJIeHx9EjdhId-RFvUig/view?usp=sharing

## Requirements
* Python 3.8.10+
* PyTorch 2.0+
* CUDA 11.4+ (★ Required for GPU acceleration)
* API for proprietary models' services (such as GPT)
<br>

## 📂 Project Structure
```text
SciChart-main/
├── ChartVLM/              # Core ChartVLM source code (tools, adapters, decoders)
├── data/                  # Dataset directory (contains .json files and /img/ subfolder)
├── example_image/         # Images used in README
├── output/                # Directory where inference results (.jsonl) are saved
├── templates/             # Prompt templates (e.g., alpaca.json used by ChartVLM)
│
├── API.py                 # Main inference script for proprietary models
├── eval_gptacc.py         # Evaluation script using GPT-based accuracy metrics
├── eval_relaxedacc.py     # Evaluation script using relaxed accuracy metrics
├── gpt_acc.py             # Helper script for GPT accuracy calculations
├── instructions.py        # Dictionary of prompts and instruction templates for models
│
├── run_chartVLM.py        # Main inference script for the ChartVLM model
├── run_Qwen-VL.py         # Main inference script for the Qwen-VL model
│
├── LICENSE                # Project license file
├── README.md              # Project documentation
└── requirements.txt       # List of Python dependencies
```

## 🛠️ Quick Start
### 1. Environment Setup
```bash
git clone https://github.com/yuetanbupt/SciChart.git
cd data
wget -O img.zip https://drive.usercontent.google.com/download?id=1M3kzYqLK26KLHJIeHx9EjdhId-RFvUig&export=download&confirm=t&uuid=d3c1008e-140e-493f-8444-b267c005f41b
unzip img.zip && rm img.zip
# Install dependencies
pip install -r requirements.txt

# Set API key (in ~/.bashrc or system environment)
export API_KEY="your_api_key_here"
```
### 2. Main Module
#### 2.1 Chart Question Answering (Proprietary Models)
Chart question-answering tasks are divided into two major categories: `BasicQA` and `ReaQA`, for which different datasets and models can be selected for evaluation respectively.
```bash
python API.py
```
🗄️ After executing the file, select a `dataset` from the `data/` folder as prompted for input. When the file finishes running, the `results` will be saved in the `output/` folder. Configure the `instructions` for specific tasks in the `instructions.py` .You can manually change the `model` used in the file. 
#### 2.2 Chart Question Answering (Open-Source Models)
For local inference using open-source vision-language models, we provide dedicated scripts for **ChartVLM** and **Qwen-VL**. These scripts also evaluate the two major categories: `BasicQA` and `ReaQA`.

⚠️ **Important Environment Requirements for ChartVLM:**
Due to the specific architecture and legacy adapters used in the official ChartVLM codebase, please ensure your environment matches the following versions to avoid compatibility errors (such as `scikit-learn` unpickling errors or `numpy` API conflicts):
```bash
# Upgrade PyTorch ecosystem to support newer accelerate features
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 accelerate==0.34.2

# Install specific versions for the adapter and model loading
pip install transformers==4.46.1 scikit-learn==1.2.2 "numpy<2.0.0"

# Install required auxiliary libraries
pip install timm einops sentencepiece protobuf fire
```
```bash
# To evaluate using the ChartVLM model
python run_chartVLM.py

# To evaluate using the Qwen-VL model
python run_Qwen-VL.py
```
🗄️ After executing the file, select a `dataset` from the `data/ `folder as prompted for input (e.g., ReaQA-EN-val). When the file finishes running, the results will be saved in the `output/` folder. Configure the instructions for specific tasks in instructions.py. Before running, please make sure to manually update the `MODEL_PATH` or `CACHE_DIR` inside the script to point to your local model weights directory.
#### 2.3 GPT-acc Evaluation
Use the method of scoring by a `LLM` to score the answers generated by the previous LLM.
```bash
python eval_gptacc.py
```
🗄️ Select the file to be scored from the `output/` folder. The final results will be directly printed in the console. You can manually change the large model used for evaluation in `gpt_acc.py`.
#### 2.4 Relaxed-acc Evaluation
Use a `rule-based` method to score the files previously generated by the LLM.
```bash
python eval_relaxedacc.py
```
🗄️ Select the file to be scored from the `output/` folder. The final results will be printed directly in the console.
#### 2.5 instructions
You can view and change the default instructions for each task configuration in the instructions.py file.
```bash
vim instructions.py
```
## 📏 Evaluation Metrics
We adopt two metrics to evaluate model performance:

**GPT-Accuracy (G-acc):** Uses GPT-4 to semantically match the model's output with the ground truth, suitable for handling diverse textual responses.
 
**Relaxed-Accuracy (R-acc):** A rule-based metric. For numerical answers (e.g., coordinates, FWHM), we allow a **5% error tolerance** to account for visual estimation capability.
## 🏆 Benchmark Results
The main results (BasicQA & ReaQA) of the SciChart.

| Model         | Open Source | G-acc | R-acc | Avg.  |
|:--------------|:-----------:|------:|------:|------:|
| Human         | -           | -     | -     | 83.84 |
| ChartVLM      | ✔           | 16.26 | 6.21  | 11.23 |
| Claude-3-haiku| ✗           | 26.88 | 14.77 | 20.83 |
| Qwen-VL       | ✔           | 25.21 | 24.83 | 25.02 |
| GPT-4o-mini   | ✗           | 33.80 | 31.65 | 32.73 |
| Gemini-pro-v  | ✗           | 41.11 | 41.22 | 41.17 |
| Gemini-2.0-T  | ✗           | 40.91 | 45.19 | 43.05 |
| GPT-4o        | ✗           | 41.93 | 47.44 | 44.68 |
| Claude-3.7-T  | ✗           | 43.64 | 46.60 | 45.12 |
## ⚠️ Limitations
**Evaluation Sensitivity:** Current metrics (including rule-based and LLM-based) can be sensitive to uncontrolled or non-standard outputs (e.g., verbose explanations when a simple number is requested).
 
**Domain Scope:** The dataset is strictly focused on scientific spectral charts and may not generalize to other types of statistical plots without further training.
## 📜 License
Our original data contributions are distributed under the MIT license.

## 🙌 Contributors and Acknowledgement
**📊 SciChart is developed by a team consisting of:**  
Tan Yue, Rui Mao, Xuzhao Shi, Zilong Song, Siyuan Xu, Yu Yan, Ziyuan Liao, Zonghai Hu, Dongyan Zhao 

WICT, Peking University  
Beijing University of Posts and Telecommunications  
Nanyang Technological University   

🤗 We sincerely appreciate the contributors of open-source annotation tools, whose work improved the efficiency and quality of our human annotation process.   

🤗 Additionally, we acknowledge the support from the State Key Laboratory of General Artificial Intelligence (SKLGAI), which provided crucial computational resources and academic guidance for this research.
## Contact Us
Email to yuetan@pku.edu.cn


