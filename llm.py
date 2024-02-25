import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a psychiatrist",
    },
    {"role": "user", "content": "I am feeling lonely, whom should I approach?"},
    {"role": "assistant", "content": "It's common to feel lonely. Tell me more about what's on your mind."},
    {"role": "user", "content": "I just moved to a new city and don't know anyone here."},
    {"role": "assistant", "content": "Moving to a new city can be challenging. How do you feel about the change?"}
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"].split("<|assistant|>")[-1])
# print(outputs)

