# Chatbot script
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np

# Choosing a model
model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for subsequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize conversation history
conversation_history = []
print("Chatbot ready! (type 'exit' to quit)\n")

while True:
    # keep only last few exchanges (prevents confusion)
    conversation_history = conversation_history[-6:]
    
    history_string = "\n".join(conversation_history)

    input_text = input("> ")
    
    # Exit condition
    if input_text.lower() == "exit":
        break

    # Tokenization of user prompt and chat history
    prompt = history_string + f"\nUser: {input_text}\nBot:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate output from the model
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.6,
        top_p=0.85
    )

    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("Bot:", response)

    # Update conversation history
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")



