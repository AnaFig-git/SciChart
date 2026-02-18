import os
import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

_cached_base_decoders = {True: None, False: None}
_cached_base_processor = None

def infer_base_decoder(image, model_path, max_token=1280, title_type=True):
    global _cached_base_decoders, _cached_base_processor
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if _cached_base_processor is None:
        _cached_base_processor = Pix2StructProcessor.from_pretrained(os.path.join(model_path,'base_decoder'))
        _cached_base_processor.image_processor.is_vqa = False

    if _cached_base_decoders[title_type] is None:
        if title_type == False:
            model = Pix2StructForConditionalGeneration.from_pretrained(os.path.join(model_path,'base_decoder'))
        else:
            model = Pix2StructForConditionalGeneration.from_pretrained(os.path.join(model_path,'base_decoder','title_type'))
        model.to(device)
        model.eval() 
        _cached_base_decoders[title_type] = model

    base_decoder = _cached_base_decoders[title_type]
    processor_base_decoder = _cached_base_processor

    inputs_base_decoder = processor_base_decoder(images=image, return_tensors="pt")
    inputs_base_decoder = inputs_base_decoder.to(device)

    with torch.no_grad():
        predictions_base_decoder = base_decoder.generate(**inputs_base_decoder, max_new_tokens=max_token)

    output_base_decoder = processor_base_decoder.decode(predictions_base_decoder[0], skip_special_tokens=True)

    return output_base_decoder
