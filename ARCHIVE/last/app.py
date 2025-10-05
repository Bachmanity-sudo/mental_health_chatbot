from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import psutil
import logging
import sys
import os
import threading

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

mem = psutil.virtual_memory()
logger.info(f"Total RAM: {mem.total / (1024**3):.2f} GB, Available: {mem.available / (1024**3):.2f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_quantization = torch.cuda.is_available()
logger.info(f"Device: {device}, Quantization: {'Enabled' if use_quantization else 'Disabled'}")

base_model_name = "t5-large"
adapter_model_name = "jblair456/t5-large-mental-health-chat-1"

# Lazy-loaded globals
tokenizer = None
base_model = None
model = None

def load_models():
    global tokenizer, base_model, model
    logger.info(f"Loading base model: {base_model_name} with adapter: {adapter_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8"
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)

    model = PeftModel.from_pretrained(base_model, adapter_model_name)
    logger.info("Model loaded successfully with LoRA adapter")

chat_history = []
MAX_HISTORY = 5
MAX_INPUT_LENGTH = 60

@app.route('/')
def index():
    return render_template('index4.html', chat_history=chat_history)

@app.route('/health')
def health():
    return "ok", 200

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if model is None or tokenizer is None:
            return jsonify({"response": "Model loading"}), 503

        if not request.is_json:
            return jsonify({"response": "Invalid request: Content-Type must be application/json"}), 400
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Please enter a message."})

        chat_history.append({"role": "user", "content": user_message})
        chat_history[:] = chat_history[-MAX_HISTORY:]

        input_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3
        )
        bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        chat_history.append({"role": "bot", "content": bot_reply.strip()})
        return jsonify({"response": bot_reply.strip()})
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    # start model load in background so the server binds immediately
    threading.Thread(target=load_models, daemon=True).start()
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
