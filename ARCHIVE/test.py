from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_id = "microsoft/Phi-3-mini-128k-instruct"
adapter_id = "jblair456/phi3-mh-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    base_id,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
    device_map=None
)

model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

question = "my dog has died and i feel sad"
prompt = f"Question: {question}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
