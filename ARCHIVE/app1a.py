from flask import Flask, request, render_template_string, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Initialize Flask app
app = Flask(__name__)

# Path to LoRA adapters and tokenizer (update to your local path)
ADAPTER_PATH = r"C:\Users\james\t5_small_merged"  # Adjust to your local folder, e.g., "C:/Users/YourName/t5_small_lora_adapter_2b"

# Load base model and tokenizer
base_id = "t5-large"
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_id)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)  # Use saved tokenizer, or "google/t5-small" if identical

# Load LoRA adapters and merge with base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()  # Merge for efficient inference
if torch.cuda.is_available():
    model = model.to("cuda")

# Chat history storage (in-memory, per session)
chat_history = []

def add_prefix(x):
    """Add task prefix if not present, as per your evaluation script."""
    xl = x.lower()
    return x if xl.startswith(("summarize:", "translate", "question:")) else f"question: {x}"

# HTML template for the chatbot UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>T5 Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }
        .chat-container { max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 20px; background: white; border-radius: 10px; }
        .chat-box { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; background: #fafafa; }
        .message { margin: 10px; padding: 10px; border-radius: 5px; }
        .user { background: #d1e7dd; text-align: right; }
        .bot { background: #f8d7da; text-align: left; }
        .input-container { display: flex; }
        input[type="text"] { flex-grow: 1; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 5px; }
        button { padding: 10px 20px; margin-left: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h2>T5 Chatbot</h2>
        <div class="chat-box" id="chat-box">
            {% for msg in chat_history %}
                <div class="message {{ 'user' if msg['role'] == 'user' else 'bot' }}">
                    <strong>{{ msg['role'].capitalize() }}:</strong> {{ msg['content'] }}
                </div>
            {% endfor %}
        </div>
        <div class="input-container">
            <input type="text" id="user-input" placeholder="Type your question..." autocomplete="off">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');

        async function sendMessage() {
            const question = userInput.value.trim();
            if (!question) return;

            // Add user message to chat
            const userDiv = document.createElement('div');
            userDiv.className = 'message user';
            userDiv.innerHTML = `<strong>User:</strong> ${question}`;
            chatBox.appendChild(userDiv);
            userInput.value = '';

            // Scroll to bottom
            chatBox.scrollTop = chatBox.scrollHeight;

            // Send question to server
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });
            const data = await response.json();

            // Add bot response to chat
            const botDiv = document.createElement('div');
            botDiv.className = 'message bot';
            botDiv.innerHTML = `<strong>Bot:</strong> ${data.response}`;
            chatBox.appendChild(botDiv);

            // Scroll to bottom
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        // Allow sending message with Enter key
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the chatbot UI with chat history."""
    return render_template_string(HTML_TEMPLATE, chat_history=chat_history)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle user question, generate response, and update chat history."""
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'response': 'Please provide a question.'})

    # Add user message to history
    chat_history.append({'role': 'user', 'content': question})

    # Prepare input with prefix
    input_text = add_prefix(question)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=50)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    gen_kwargs = dict(
    max_new_tokens=256,
    min_new_tokens=64,
    length_penalty=1.2,
    num_beams=6,
    top_p=0.90,
    top_k=40,
    temperature=0.9,
    no_repeat_ngram_size=3,
)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Add bot response to history
    chat_history.append({'role': 'bot', 'content': response})

    # Limit history to avoid memory issues (e.g., keep last 20 messages)
    if len(chat_history) > 20:
        chat_history[:] = chat_history[-20:]

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)