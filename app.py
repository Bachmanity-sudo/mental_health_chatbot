from flask import Flask, request, render_template, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Initialize Flask app
app = Flask(__name__)

# Hugging Face LoRA adapter repo
MODEL_NAME = "jblair456/t5-large-mental-health-chat-1"

# Load base model and tokenizer
base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load LoRA adapters and merge with base model
model = PeftModel.from_pretrained(base_model, MODEL_NAME)
model = model.merge_and_unload()  # Merge for efficient inference
if torch.cuda.is_available():
    model = model.to("cuda")

# Chat history storage (in-memory, per session)
chat_history = []

def add_prefix(x):
    xl = x.lower()
    return x if xl.startswith(("summarize:", "translate", "question:")) else f"question: {x}"

@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'response': 'Please provide a question.'})

    chat_history.append({'role': 'user', 'content': question})

    input_text = add_prefix(question)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=50)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=256,
        min_new_tokens=64,
        length_penalty=1.2,
        num_beams=6,
        top_p=0.98,
        top_k=100,
        temperature=0.9,
        no_repeat_ngram_size=3,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    chat_history.append({'role': 'bot', 'content': response})
    if len(chat_history) > 20:
        chat_history[:] = chat_history[-20:]

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
