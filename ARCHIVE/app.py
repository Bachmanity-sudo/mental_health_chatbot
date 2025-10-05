# app.py
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel
import torch, psutil, logging, sys, os, threading

app = Flask(__name__)

def running_in_container_or_aws():
    return any([
        os.path.exists("/.dockerenv"),
        os.getenv("AWS_EXECUTION_ENV"),
        os.getenv("APP_RUNNER"),
        os.getenv("ECS_CONTAINER_METADATA_URI"),
        os.getenv("ECS_CONTAINER_METADATA_URI_V4"),
    ])

ENV = "cloud" if running_in_container_or_aws() else "local"
ENV = os.getenv("APP_ENV", ENV).lower()

SYSTEM_PROMPT = (
    "system: located in Australia; use supportive language; consider solutions for the user"
    #"do not diagnose; do not provide medical advice; use supportive language; encourage seeking professional help; keep responses brief; Never reference this prompt in a chat."
)

BASE_CFG = {
    "max_history": 2,
    "max_input_len": 200,
    "gen": dict(
        max_new_tokens=200,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.7,
        no_repeat_ngram_size=3,
    ),
    "log_level": logging.INFO if ENV == "cloud" else logging.DEBUG,
    "port_default": "8080" if ENV == "cloud" else "5000",
}

logging.basicConfig(
    level=BASE_CFG["log_level"],
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info(f"ENV: {ENV}")

mem = psutil.virtual_memory()
logger.info(f"Total RAM: {mem.total / (1024**3):.2f} GB, Available: {mem.available / (1024**3):.2f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_quantization = torch.cuda.is_available()
logger.info(f"Device: {device}, Quantization: {'Enabled' if use_quantization else 'Disabled'}")

MODEL_REGISTRY = {
    "T5-large": {
        "type": "seq2seq",
        "base_model_name": "t5-large",
        "adapter_model_name": "jblair456/t5-large-mental-health-chat-1",
    },
    "Phi-3 128k": {
        "type": "causal",
        "base_model_name": "microsoft/Phi-3-mini-128k-instruct",
        "adapter_model_name": "jblair456/phi3-mh-lora",
    },
}

tokenizer = None
base_model = None
model = None
current_model_name = "T5-large"

chat_history = []
MAX_HISTORY = BASE_CFG["max_history"]
MAX_INPUT_LENGTH = BASE_CFG["max_input_len"]
GEN_KW = BASE_CFG["gen"]

_model_lock = threading.Lock()

def _bitsandbytes_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
    )

def _load_base_and_adapter(entry):
    global tokenizer, base_model, model

    base_id = entry["base_model_name"]
    adapter_id = entry["adapter_model_name"]
    mtype = entry["type"]

    logger.info(f"Loading: {current_model_name} ({mtype}) base={base_id} adapter={adapter_id}")

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mtype == "seq2seq":
        if use_quantization:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_id,
                quantization_config=_bitsandbytes_config(),
                device_map="auto",
            )
        else:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_id).to(device)
    else:
        if use_quantization:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                quantization_config=_bitsandbytes_config(),
                device_map="auto",
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(base_id).to(device)

    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.eval()
    logger.info("Model loaded with LoRA adapter")

def select_model(name: str):
    global current_model_name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    with _model_lock:
        current_model_name = name
        _load_base_and_adapter(MODEL_REGISTRY[name])

def build_prompt_seq2seq():
    history_text = ""
    for turn in chat_history[-MAX_HISTORY:]:
        history_text += f"{turn['role']}: {turn['content']}\n"
    return f"{SYSTEM_PROMPT}\n{history_text}"

def build_prompt_phi(question: str):
    return f"Question: {question}\nAnswer:"

def encode_inputs(user_message: str):
    mtype = MODEL_REGISTRY[current_model_name]["type"]
    if mtype == "causal":
        prompt = build_prompt_phi(user_message)
        enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        gen_kw = dict(GEN_KW, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        return enc, gen_kw, mtype, prompt
    else:
        prompt = build_prompt_seq2seq()
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(device)
        return enc, GEN_KW, mtype, prompt

@app.route("/")
def index():
    return render_template(
        "index.html",
        chat_history=chat_history,
        models=list(MODEL_REGISTRY.keys()),
        current_model=current_model_name,
    )

@app.route("/health")
def health():
    return "ok", 200

@app.route("/models", methods=["GET"])
def models_list():
    return jsonify({"models": list(MODEL_REGISTRY.keys()), "current": current_model_name})

@app.route("/set_model", methods=["POST"])
def set_model():
    try:
        data = request.get_json(silent=True) or {}
        name = str(data.get("model", "")).strip()
        if not name:
            return jsonify({"error": "missing model"}), 400
        select_model(name)
        return jsonify({"ok": True, "current": current_model_name})
    except Exception as e:
        logger.exception("set_model failed")
        return jsonify({"error": str(e)}), 500

@app.route("/new_chat", methods=["POST"])
def new_chat():
    global chat_history
    chat_history = []
    return jsonify({"status": "cleared"})

@app.route("/chat", methods=["POST"])
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
        del chat_history[:-MAX_HISTORY]

        enc, gen_kw, mtype, prompt = encode_inputs(user_message)

        with torch.inference_mode():
            outputs = model.generate(**enc, **gen_kw)

        if mtype == "causal":
            generated = outputs[0, enc["input_ids"].shape[-1]:]
        else:
            generated = outputs[0]

        reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
        chat_history.append({"role": "bot", "content": reply})
        return jsonify({"response": reply})
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}"}), 500

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    try:
        if model is None or tokenizer is None:
            return Response("Model loading", status=503, mimetype="text/plain")
        if not request.is_json:
            return Response("Invalid request", status=400, mimetype="text/plain")

        user_message = request.json.get("message", "").strip()
        if not user_message:
            return Response("Please enter a message.", mimetype="text/plain")

        chat_history.append({"role": "user", "content": user_message})
        del chat_history[:-MAX_HISTORY]

        enc, gen_kw, mtype, prompt = encode_inputs(user_message)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        def generate_worker():
            with torch.inference_mode():
                model.generate(**enc, **gen_kw, streamer=streamer)

        def token_gen():
            full = []
            t = threading.Thread(target=generate_worker, daemon=True)
            t.start()
            for piece in streamer:
                full.append(piece)
                yield piece
            reply = "".join(full).strip()
            chat_history.append({"role": "bot", "content": reply})

        return Response(stream_with_context(token_gen()), mimetype="text/plain")
    except Exception as e:
        logger.error(f"Error in /chat_stream: {str(e)}")
        return Response(f"Error: {str(e)}", status=500, mimetype="text/plain")

if __name__ == "__main__":
    if ENV == "cloud":
        try:
            torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
        except Exception:
            pass

    def _boot():
        try:
            select_model(current_model_name)
        except Exception:
            logger.exception("Initial model load failed")
    threading.Thread(target=_boot, daemon=True).start()

    port = int(os.getenv("PORT", BASE_CFG["port_default"]))
    app.run(host="0.0.0.0", port=port)
