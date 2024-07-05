from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import copy
import os
import time
from pprint import pprint

from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

image = Image.open("/home/gsfc-pi/dev/media/selfie_new.jpeg")

start = time.perf_counter()
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
    model_id = 'microsoft/Florence-2-base'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print(f"Model loading time - {time.perf_counter() - start:.2f} seconds")

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer

print("running simple caption")

start = time.perf_counter()
response = run_example('<CAPTION>')
pprint(f"Simple caption response - {response}")
print(f"Inference time - {time.perf_counter() - start:.2f} seconds")

print("------")
print("\n")

print("running detailed caption")

start = time.perf_counter()
response = run_example('<DETAILED_CAPTION>')
pprint(f"Detailed caption response - {response}")
print(f"Inference time - {time.perf_counter() - start:.2f} seconds")

print("------")
print("\n")

print("running more detailed caption")

start = time.perf_counter()
response = run_example('<MORE_DETAILED_CAPTION>')
pprint(f"More Detailed caption response - {response}")
print(f"Inference time - {time.perf_counter() - start:.2f} seconds")

print("------")
print("\n")



