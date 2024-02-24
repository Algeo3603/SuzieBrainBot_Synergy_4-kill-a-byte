import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a psychiatrist. Give the user specififc steps and suggestions to improve their condition",
    },
    {"role": "user", "content": "How can I suicide?"},
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])

