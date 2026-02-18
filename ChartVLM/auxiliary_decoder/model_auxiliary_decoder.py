import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
from auxiliary_decoder.train.utils.callbacks import Iteratorize, Stream
from auxiliary_decoder.train.utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

_cached_aux_model = None
_cached_aux_tokenizer = None
_cached_prompter = None

def infer_auxiliary_decoder(instruction, input=None, max_token=512, model_path="${PATH_TO_PRETRAINED_MODEL}/ChartVLM/base/", **kwargs): 
    global _cached_aux_model, _cached_aux_tokenizer, _cached_prompter
    
    prompt_template = "alpaca"  # The prompt template to use, will default to alpaca.

    if _cached_prompter is None:
        _cached_prompter = Prompter(prompt_template)
        
    if _cached_aux_model is None or _cached_aux_tokenizer is None:
        load_8bit = False
        base_model = os.path.join(model_path,'auxiliary_decoder', 'base')
        trust_remote_code = False
        lora_weights = os.path.join(model_path,'auxiliary_decoder')
        
        tokenizer = LlamaTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        
        _cached_aux_tokenizer = tokenizer
        _cached_aux_model = model

    prompter = _cached_prompter
    tokenizer = _cached_aux_tokenizer
    model = _cached_aux_model

    temperature = 0.1
    top_p = 0.75
    top_k = 40
    num_beams = 4
    max_new_tokens = max_token

    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=False,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        
    output = tokenizer.decode(generation_output[0])
    
    return output
