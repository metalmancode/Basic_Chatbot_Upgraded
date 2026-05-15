from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

warnings.filterwarnings("ignore")

# Step 2: Choose a modern LLM
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Step 3: Load model and tokenizer
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.unk_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32
)

# Step 4: Initialize conversation messages
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Give short and concise answers in 2-3 lines."
    }
]

# Step 5: Start chatbot loop
print("Chatbot started. Type 'exit' to quit.\n")
while True:
    user_input = input("> ")

    if user_input.lower() == "exit":
        break

    # Step 5.1: Update conversation history
    messages.append({"role": "user", "content": user_input})

    # keep only last few exchanges (prevents confusion)
    messages = [messages[0]] + messages[-10:]

    # Step 5.2 : Apply chat template
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        max_length=512
    )

    # Step 5.3: Generate response
    with torch.inference_mode():
        outputs = model.generate(
            tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=60,
            temperature=0.5,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id
        )

    # Step 5.4: Decode and display response
    response = tokenizer.decode(
        outputs[0][tokenized["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    print(f"Bot: {response}\n")

    # Step 5.5 : Save assistant response
    messages.append({"role": "assistant", "content": response})
