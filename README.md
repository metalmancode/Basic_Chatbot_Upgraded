# Basic AI Chatbot using Transformers

A lightweight, terminal-based chatbot built with Python and Hugging Face's `transformers` library. This project uses the **Blenderbot-400M-distill** model, which is optimized for natural dialogue and can run efficiently on a CPU.

## 🚀 Features

- **Open Source LLM**: Powered by Facebook's Blenderbot.
- **Contextual Memory**: Remembers the last few exchanges to provide coherent responses.
- **Natural Generation**: Uses sampling, temperature control, and repetition penalties for human-like conversation.
- **CPU Optimized**: Lightweight enough to run without a high-end GPU.
- **Virtual Environment Ready**: Easy to set up and run in an isolated environment.

## 🛠️ Technology Stack

- **Python 3.x**
- **Transformers**: For loading the model and tokenizer.
- **PyTorch**: Backend for model execution.
- **NumPy**: Data handling.

## 📋 Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/metalmancode/basic_chatbot.git
   cd basic_chatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv my_env
   source my_env/bin/activate  # On Windows use: my_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install transformers torch numpy accelerate
   ```

## 🏃 Usage

Run the chatbot using the following command:

```bash
python chatbot.py
```

- Type your message at the `>` prompt.
- Type `exit` to end the conversation.

## 🤖 Upgraded Version: Modern LLM Chatbot (SmolLM2)

The upgraded version (`chatbot_llm.py`) uses **SmolLM2-360M-Instruct**, a state-of-the-art causal language model. This version introduces **Chat Templates**, which handle the complex formatting of "system", "user", and "assistant" roles automatically.

### Key Upgrades:
- **Causal LLM**: Uses `AutoModelForCausalLM` for superior reasoning and text generation.
- **System Prompting**: Allows defining the bot's behavior (e.g., "be concise").
- **Chat Templates**: Uses `tokenizer.apply_chat_template` to format messages correctly for the model.
- **Inference Mode**: Optimized with `torch.inference_mode()` for faster CPU performance.

## 🏃 Running the Upgraded Bot

```bash
python chatbot_llm.py
```

## 🧠 How the Upgraded Bot Works

1. **Structured Messages**: Uses a list of dictionaries with `role` and `content`.
2. **Template Application**: Converts messages into a single string with special tokens that the model was trained on.
3. **Selective Decoding**: Only decodes the newly generated tokens, ensuring the bot doesn't repeat the user's input.


## 📝 License

This project is open-source and available under the MIT License.
