# app3.py
import os, sys, json, time, uuid, threading, logging
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, Response,
    stream_with_context, send_from_directory, abort, url_for
)
from jinja2 import TemplateNotFound
import torch, psutil
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

from review_output import review_output

app = Flask(__name__)

def running_in_container_or_aws():
    return any([
        os.path.exists("/.dockerenv"),
        os.getenv("AWS_EXECUTION_ENV"),
        os.getenv("APP_RUNNER"),
        os.getenv("ECS_CONTAINER_METADATA_URI"),
        os.getenv("ECS_CONTAINER_METADATA_URI_V4"),
    ])

ENV = os.getenv("APP_ENV", "cloud" if running_in_container_or_aws() else "local").lower()

SYSTEM_PROMPT = "system: located in Australia; use supportive language; consider solutions for the user"

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
    "tz_local": os.getenv("TZ_LOCAL", "Australia/Perth"),
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
    "T5-large (FT)": {
        "type": "seq2seq",
        "base_model_name": "t5-large",
        "adapter_model_name": "jblair456/t5-large-mental-health-chat-1",
    },
    "T5-large (base)": {
        "type": "seq2seq",
        "base_model_name": "t5-large",
        "adapter_model_name": None,
    },
    "Phi-3 128k (FT)": {
        "type": "causal",
        "base_model_name": "microsoft/Phi-3-mini-128k-instruct",
        "adapter_model_name": "jblair456/phi3-mh-lora",
    },
    "T5-base (FT)": {
        "type": "seq2seq",
        "base_model_name": "t5-base",
        "adapter_model_name": "jblair456/t5-base-mental-health-chat-4",
    },
}

tokenizer = None
base_model = None
model = None
current_model_name = "T5-large (FT)"

chat_history = []
MAX_HISTORY = BASE_CFG["max_history"]
MAX_INPUT_LENGTH = BASE_CFG["max_input_len"]
GEN_KW = BASE_CFG["gen"]

_model_lock = threading.Lock()
_session_lock = threading.Lock()
_messages_lock = threading.Lock()
current_session_id = None
turn_index = 0

# ---- paths and simple message store in root/messages.json ----
BASE_DIR = Path(__file__).resolve().parent
MESSAGES_PATH = BASE_DIR / "messages.json"

def _atomic_write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)

def init_logs_dir():
    with _messages_lock:
        if not MESSAGES_PATH.exists():
            _atomic_write_json(MESSAGES_PATH, {"messages": []})

def _read_messages_file():
    if not MESSAGES_PATH.exists():
        return []
    try:
        with open(MESSAGES_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return []
        try:
            obj = json.loads(raw)
            return obj.get("messages", obj if isinstance(obj, list) else [])
        except json.JSONDecodeError:
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
    except Exception:
        logger.exception("read messages.json failed")
        return []

def _append_message_record(rec: dict):
    with _messages_lock:
        msgs = _read_messages_file()
        msgs.append(rec)
        _atomic_write_json(MESSAGES_PATH, {"messages": msgs})

def create_session(session_id: str, tz_local: str):
    _append_message_record({
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "session_start",
        "session_id": session_id,
        "tz_local": tz_local,
    })

def append_turn_record(session_id: str, record: dict):
    _append_message_record(dict(record))

def update_session_index_row(session_id: str, **kwargs):
    # No separate index file; optional no-op
    pass

def _find_messages_file():
    return str(MESSAGES_PATH) if MESSAGES_PATH.exists() else None

def load_messages_with_logging():
    path = _find_messages_file()
    msgs, err = [], None
    if not path:
        err = "messages.json not found"
        return msgs, err, None
    try:
        msgs = _read_messages_file()
    except Exception as e:
        err = f"read/parse error: {e}"
        logger.exception("Sessions: %s", err)
    return msgs, err, path

def _load_messages():
    return load_messages_with_logging()

# ---- model loading ----
def _bitsandbytes_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

def _load_base_and_adapter(entry):
    global tokenizer, base_model, model
    base_id = entry["base_model_name"]
    adapter_id = entry.get("adapter_model_name")
    mtype = entry["type"]
    logger.info(f"Loading: {current_model_name} ({mtype}) base={base_id} adapter={adapter_id}")

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def _load_seq2seq():
        if use_quantization:
            try:
                return AutoModelForSeq2SeqLM.from_pretrained(
                    base_id, quantization_config=_bitsandbytes_config(), device_map="auto"
                )
            except Exception:
                logger.warning("Quantized seq2seq load failed; falling back")
        return AutoModelForSeq2SeqLM.from_pretrained(base_id, low_cpu_mem_usage=True).to(device)

    def _load_causal():
        if use_quantization:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    base_id, quantization_config=_bitsandbytes_config(), device_map="auto"
                )
            except Exception:
                logger.warning("Quantized causal load failed; falling back")
        return AutoModelForCausalLM.from_pretrained(base_id, low_cpu_mem_usage=True).to(device)

    base_model = _load_seq2seq() if mtype == "seq2seq" else _load_causal()
    model = PeftModel.from_pretrained(base_model, adapter_id) if adapter_id else base_model
    model.eval()
    logger.info("Model ready")

def select_model(name: str):
    global current_model_name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    with _model_lock:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        gen_kw = dict(
            GEN_KW,
            eos_token_id=getattr(model.config, "eos_token_id", tokenizer.eos_token_id),
            pad_token_id=tokenizer.pad_token_id,
        )
        return enc, gen_kw, mtype, prompt

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _ensure_session():
    global current_session_id, turn_index
    with _session_lock:
        if not current_session_id:
            sid = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}_{uuid.uuid4().hex[:8]}"
            current_session_id = sid
            turn_index = 0
            create_session(sid, BASE_CFG["tz_local"])
        return current_session_id

def _log_turn(user_message: str, reply: str, mtype: str, prompt: str, gen_kw: dict, lat_ms: float):
    global turn_index
    turn_index += 1
    try:
        metrics = review_output(reply)
    except Exception:
        logger.exception("review_output failed")
        metrics = {}
    record = {
        "ts_utc": _utc_now_iso(),
        "session_id": current_session_id,
        "turn": turn_index,
        "model_name": current_model_name,
        "model_type": mtype,
        "latency_ms": int(lat_ms),
        "gen_params": gen_kw,
        "prompt_preview": prompt[:200],
        "user_message": user_message,
        "response_preview": reply[:500],
        "metrics": metrics,
    }
    try:
        append_turn_record(current_session_id, record)
        update_session_index_row(
            current_session_id,
            last_ts_utc=record["ts_utc"],
            model_name=current_model_name,
            decision=metrics.get("decision"),
        )
    except Exception:
        logger.exception("log write failed")

# ---- views ----
@app.route("/")
def index():
    try:
        return render_template("index.html", chat_history=chat_history, models=list(MODEL_REGISTRY.keys()), current_model=current_model_name)
    except TemplateNotFound:
        return Response("templates/index.html missing", status=500, mimetype="text/plain")

@app.get("/sessions.html")
def sessions_html():
    logger.info(f"/sessions.html list {MESSAGES_PATH}")
    files = [MESSAGES_PATH.name] if MESSAGES_PATH.exists() else []
    return render_template(
        "sessions.html",
        files=files,
        home_url="/",
        cwd=str(BASE_DIR),
        sessions_dir=str(BASE_DIR),
        messages_path=str(MESSAGES_PATH),
        files_url=url_for("list_session_files"),
        messages_api=url_for("serve_messages_json"),
    )

@app.get("/logs/files")
def list_session_files():
    files = [MESSAGES_PATH.name] if MESSAGES_PATH.exists() else []
    return jsonify(files)

@app.get("/logs/sessions/<path:filename>")
def serve_session_file(filename):
    if filename != MESSAGES_PATH.name:
        abort(404)
    return send_from_directory(str(BASE_DIR), filename)

@app.get("/logs/messages")
@app.get("/messages")
def serve_messages_json():
    msgs, err, path = load_messages_with_logging()
    if err:
        logger.error("messages.json load failed: %s", err)
        status = 404 if "not found" in err.lower() else 500
        payload = {"error": err}
        if path:
            payload["path"] = path
        return jsonify(payload), status
    return jsonify(msgs)

@app.route("/health")
def health():
    return "ok", 200

# ---- model mgmt ----
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
    global chat_history, current_session_id, turn_index
    chat_history = []
    current_session_id = None
    turn_index = 0
    return jsonify({"status": "cleared"})

# ---- chat ----
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

        _ensure_session()

        chat_history.append({"role": "user", "content": user_message})
        del chat_history[:-MAX_HISTORY]

        enc, gen_kw, mtype, prompt = encode_inputs(user_message)
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**enc, **gen_kw)
        lat_ms = (time.perf_counter() - t0) * 1000

        if mtype == "causal":
            generated = outputs[0, enc["input_ids"].shape[-1]:]
        else:
            generated = outputs[0]

        reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
        chat_history.append({"role": "bot", "content": reply})

        _log_turn(user_message, reply, mtype, prompt, gen_kw, lat_ms)
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

        _ensure_session()

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
            t0 = time.perf_counter()
            try:
                for piece in streamer:
                    full.append(piece)
                    yield piece
            finally:
                reply = "".join(full).strip()
                chat_history.append({"role": "bot", "content": reply})
                lat_ms = (time.perf_counter() - t0) * 1000
                try:
                    _log_turn(user_message, reply, mtype, prompt, gen_kw, lat_ms)
                except Exception:
                    logger.exception("_log_turn failed")

        return Response(stream_with_context(token_gen()), mimetype="text/plain")
    except Exception as e:
        logger.error(f"Error in /chat_stream: {str(e)}")
        return Response(f"Error: {str(e)}", status=500, mimetype="text/plain")

if __name__ == "__main__":
    init_logs_dir()
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
