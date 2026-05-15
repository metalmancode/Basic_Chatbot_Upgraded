from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
print("Loading model for Web UI...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.unk_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32
)

# In-memory session storage (per-user history would need a real database or session)
conversation_history = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history
    data = request.json
    user_input = data.get("message")
    
    # Get parameters from UI
    temperature = float(data.get("temperature", 0.5))
    top_p = float(data.get("top_p", 0.8))
    max_new_tokens = int(data.get("max_tokens", 60))
    system_prompt = data.get("system_prompt", "You are a helpful AI assistant.")

    # Reset history if requested or if system prompt changed
    if data.get("reset"):
        conversation_history = [{"role": "system", "content": system_prompt}]
    
    # Update system prompt if it's different
    if conversation_history[0]["content"] != system_prompt:
        conversation_history[0]["content"] = system_prompt

    # Add user message
    conversation_history.append({"role": "user", "content": user_input})
    
    # Keep context window
    context = [conversation_history[0]] + conversation_history[-10:]

    # Apply template
    tokenized = tokenizer.apply_chat_template(
        context,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        max_length=512
    )

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(
        outputs[0][tokenized["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # Save assistant response
    conversation_history.append({"role": "assistant", "content": response})

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
