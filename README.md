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

## 🧠 How It Works

1. **Model Selection**: Uses `facebook/blenderbot-400M-distill`, a sequence-to-sequence model fine-tuned for conversation.
2. **Tokenization**: The `AutoTokenizer` converts user text into numerical tokens that the model understands.
3. **Prompt Engineering**: The script combines conversation history with the new user input, formatted with `User:` and `Bot:` labels to guide the model.
4. **Generation Parameters**:
   - `max_new_tokens=60`: Limits response length.
   - `temperature=0.6`: Balances creativity and coherence.
   - `repetition_penalty=1.3`: Prevents the bot from repeating the same words.
5. **Context Window**: Keeps the last 6 exchanges to maintain relevant context without exceeding token limits.

## 📝 License

This project is open-source and available under the MIT License.
