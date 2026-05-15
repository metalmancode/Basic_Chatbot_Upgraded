# AI Chatbot Pro - Upgraded (SmolLM2 + Flask)

A state-of-the-art, web-based conversational AI application built with **SmolLM2-360M-Instruct** and **Flask**. This project represents an "Upgraded" version of the basic transformer chatbot, featuring a premium web interface, dynamic parameter tuning, and advanced error handling.

## 🌟 Key Features

### 1. Modern Causal LLM
- **Model**: `HuggingFaceTB/SmolLM2-360M-Instruct`.
- **Architecture**: A compact but powerful instruction-tuned causal language model.
- **Chat Templates**: Automatically handles system, user, and assistant roles using the model's native template format.

### 2. Premium Web Interface
- **Glassmorphism UI**: A sleek, dark-mode design built with modern CSS.
- **Dynamic Control Sidebar**: Adjust AI parameters (Temperature, Top P, Max Tokens) in real-time.
- **Responsive Design**: Works perfectly across different screen sizes.

### 3. Robust Error Handling
- **Backend Protection**: Python `try-except` blocks ensure the server stays up even during model generation errors.
- **User-Friendly Alerts**: Custom UI components to inform users about connection issues or generation failures.

### 4. Context Management
- **Sliding Window Memory**: Keeps the last 10 exchanges for context-aware replies without overwhelming the model's token limit.

---

## 🛠️ Technology Stack

- **Backend**: Python, Flask, PyTorch
- **AI Library**: Hugging Face Transformers
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+)
- **Model Engine**: SmolLM2 (CPU Optimized)

---

## ⚙️ Installation & Setup

### 📋 Prerequisites
- Python 3.8+
- Git

### 🚀 Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/metalmancode/Basic_Chatbot_Upgraded.git
   cd Basic_Chatbot_Upgraded
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Application

### 1. Web Interface (Recommended)
Run the Flask server:
```bash
python app.py
```
Open [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser.

### 2. Terminal Mode
If you prefer a lightweight terminal experience:
```bash
python chatbot_llm.py
```

---

## 🧠 Advanced Configuration

### Understanding the Parameters
- **System Prompt**: Set the "personality" of the bot. Tell it to be "concise", "creative", or a "professional assistant".
- **Temperature (0.1 - 1.5)**: 
  - Lower (e.g., 0.2): More predictable and factual.
  - Higher (e.g., 0.8): More creative and diverse.
- **Top P (0.1 - 1.0)**: Nucleus sampling; keeps the cumulative probability of word choices within a certain range.
- **Max Tokens**: Limits the length of the bot's response.

---

## 🔧 Troubleshooting & Error Handling

- **Connection Error**: Ensure the Flask server is running in your terminal.
- **Generation Error**: If the bot returns a red error message, try lowering the `Max Tokens` or clicking "Clear History" to reset the context window.
- **Slow Responses**: Since this runs on the CPU, generation speed depends on your local hardware. Close other resource-heavy apps for better performance.

---

## 📝 License
This project is open-source and available under the **MIT License**.

---

## 👨‍💻 Author
Developed by **metalmancode**.
