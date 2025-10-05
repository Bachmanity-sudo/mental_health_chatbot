# app.py
import os, sys, json, time, uuid, threading, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import (
    Flask, render_template, request, jsonify, Response,
    stream_with_context, url_for
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
    "port_default": "5000",
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
current_session_id = None
turn_index = 0

# ---- chat session storage ----
BASE_DIR = Path(__file__).resolve().parent

class MessageStoreError(Exception):
    """Raised when the chat log store cannot complete an operation."""


class LocalMessageStore:
    def __init__(self, base_dir: Path):
        self._path = base_dir / "messages.json"
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        with self._lock:
            if not self._path.exists():
                self._atomic_write({"messages": []})

    def _atomic_write(self, data):
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self._path)

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                return []
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj.get("messages", [])
                if isinstance(obj, list):
                    return obj
            except json.JSONDecodeError:
                return [json.loads(line) for line in raw.splitlines() if line.strip()]
        except Exception:
            logger.exception("read messages.json failed")
        return []

    def _write_all(self, messages: List[Dict[str, Any]]):
        self._atomic_write({"messages": messages})

    def create_session(self, session_id: str, tz_local: str) -> None:
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": "session_start",
            "session_id": session_id,
            "tz_local": tz_local,
        }
        with self._lock:
            msgs = self._read_all()
            msgs.append(record)
            self._write_all(msgs)

    def append_turn_record(self, session_id: str, record: dict) -> None:
        with self._lock:
            msgs = self._read_all()
            msgs.append(dict(record))
            self._write_all(msgs)

    def update_session_index_row(self, session_id: str, **kwargs) -> None:
        return None

    def load_all_messages(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._read_all())

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        return [m for m in self.load_all_messages() if m.get("session_id") == session_id]

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions: Dict[str, Dict[str, Any]] = {}
        for item in self.load_all_messages():
            sid = item.get("session_id")
            if not sid:
                continue
            record = sessions.setdefault(sid, {
                "session_id": sid,
                "turns": 0,
                "created_ts": item.get("ts_utc"),
                "last_ts": item.get("ts_utc"),
                "tz_local": item.get("tz_local"),
            })
            ts = item.get("ts_utc")
            if ts:
                prev_last = record.get("last_ts")
                if not prev_last or ts > prev_last:
                    record["last_ts"] = ts
                if item.get("event") == "session_start":
                    record["created_ts"] = ts
            if item.get("event") != "session_start":
                record["turns"] = record.get("turns", 0) + 1
            if item.get("model_name"):
                record["model_name"] = item.get("model_name")
        return sorted(sessions.values(), key=lambda r: r.get("last_ts") or "", reverse=True)

    def describe(self) -> Dict[str, Any]:
        return {"backend": "local", "location": str(self._path)}


class S3MessageStore:
    def __init__(self, bucket: str, prefix: Optional[str] = None):
        try:
            import boto3  # type: ignore
            from botocore.config import Config as BotoConfig  # type: ignore
            from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
        except ImportError as exc:
            raise MessageStoreError("boto3 is required for S3 storage") from exc

        self.bucket = bucket
        self.prefix = prefix.strip().strip("/") if prefix else ""
        base = f"{self.prefix}/" if self.prefix else ""
        self.sessions_prefix = f"{base}sessions/"
        self._client = boto3.client("s3", config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}))
        self._client_error = ClientError
        self._botocore_error = BotoCoreError

    def ensure_ready(self) -> None:
        return None

    def _now_ts(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _event_key(self, session_id: str, ts: Optional[str], suffix: str) -> str:
        base_ts = ts or self._now_ts()
        safe_ts = base_ts.replace(":", "-")
        return f"{self.sessions_prefix}{session_id}/{safe_ts}_{suffix}.json"

    def _put_json(self, key: str, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")
        except (self._client_error, self._botocore_error) as exc:
            raise MessageStoreError(f"S3 put_object failed for {key}: {exc}") from exc

    def _read_json(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            obj = self._client.get_object(Bucket=self.bucket, Key=key)
        except self._client_error as exc:
            if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
                return None
            raise MessageStoreError(f"S3 get_object failed for {key}: {exc}") from exc
        except self._botocore_error as exc:
            raise MessageStoreError(f"S3 get_object failed for {key}: {exc}") from exc
        body = obj.get("Body")
        if not body:
            return None
        data = body.read().decode("utf-8")
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in S3 object %s", key)
            return None

    def _meta_key(self, session_id: str) -> str:
        return f"{self.sessions_prefix}{session_id}/meta.json"

    def _load_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._read_json(self._meta_key(session_id))

    def _write_meta(self, session_id: str, meta: Dict[str, Any]) -> None:
        self._put_json(self._meta_key(session_id), meta)

    def create_session(self, session_id: str, tz_local: str) -> None:
        ts = self._now_ts()
        record = {
            "ts_utc": ts,
            "event": "session_start",
            "session_id": session_id,
            "tz_local": tz_local,
        }
        key = self._event_key(session_id, ts, f"start-{uuid.uuid4().hex[:8]}")
        self._put_json(key, record)
        meta = {
            "session_id": session_id,
            "created_ts": ts,
            "last_ts": ts,
            "tz_local": tz_local,
            "turns": 0,
        }
        self._write_meta(session_id, meta)

    def append_turn_record(self, session_id: str, record: dict) -> None:
        record = dict(record)
        ts = record.get("ts_utc") or self._now_ts()
        key = self._event_key(session_id, ts, f"turn-{int(record.get('turn', 0)):04d}-{uuid.uuid4().hex[:8]}")
        self._put_json(key, record)
        meta = self._load_meta(session_id) or {
            "session_id": session_id,
            "created_ts": ts,
            "tz_local": record.get("tz_local"),
            "turns": 0,
            "last_ts": ts,
        }
        meta["last_ts"] = ts
        meta["turns"] = meta.get("turns", 0) + 1
        if record.get("model_name"):
            meta["model_name"] = record.get("model_name")
        self._write_meta(session_id, meta)

    def update_session_index_row(self, session_id: str, **kwargs) -> None:
        meta = self._load_meta(session_id)
        if not meta:
            return None
        meta.update({k: v for k, v in kwargs.items() if v is not None})
        if kwargs.get("last_ts_utc"):
            meta["last_ts"] = kwargs["last_ts_utc"]
        self._write_meta(session_id, meta)
        return None

    def _list_session_prefixes(self) -> List[str]:
        prefixes: List[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.sessions_prefix, Delimiter='/'):
                for pref in page.get("CommonPrefixes", []):
                    key = pref.get("Prefix")
                    if key and key.startswith(self.sessions_prefix):
                        prefixes.append(key)
        except (self._client_error, self._botocore_error) as exc:
            raise MessageStoreError(f"list_objects_v2 failed: {exc}") from exc
        return prefixes

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        for prefix in self._list_session_prefixes():
            session_id = prefix[len(self.sessions_prefix):-1]
            meta = self._load_meta(session_id)
            if meta:
                sessions.append(meta)
            else:
                sessions.append({"session_id": session_id})
        sessions.sort(key=lambda r: r.get("last_ts") or r.get("created_ts") or "", reverse=True)
        return sessions

    def _list_session_event_keys(self, session_id: str) -> List[str]:
        prefix = f"{self.sessions_prefix}{session_id}/"
        keys: List[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if key and not key.endswith("meta.json"):
                        keys.append(key)
        except (self._client_error, self._botocore_error) as exc:
            raise MessageStoreError(f"list_objects_v2 failed for {session_id}: {exc}") from exc
        keys.sort()
        return keys

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for key in self._list_session_event_keys(session_id):
            payload = self._read_json(key)
            if isinstance(payload, dict):
                messages.append(payload)
        messages.sort(key=lambda r: r.get("ts_utc") or "")
        return messages

    def load_all_messages(self) -> List[Dict[str, Any]]:
        combined: List[Dict[str, Any]] = []
        for session in self.list_sessions():
            sid = session.get("session_id")
            if not sid:
                continue
            combined.extend(self.get_session_messages(sid))
        combined.sort(key=lambda r: r.get("ts_utc") or "")
        return combined

    def describe(self) -> Dict[str, Any]:
        info = {"backend": "s3", "bucket": self.bucket}
        if self.prefix:
            info["prefix"] = self.prefix
        return info


def _build_message_store() -> Any:
    bucket = os.getenv("CHATLOG_S3_BUCKET", "").strip()
    prefix = os.getenv("CHATLOG_S3_PREFIX", "").strip()
    if bucket:
        try:
            store = S3MessageStore(bucket=bucket, prefix=prefix)
            logger.info("Using S3 message store bucket=%s prefix=%s", bucket, prefix or "<root>")
            return store
        except MessageStoreError:
            logger.exception("Falling back to local message store after S3 init failure")
    store = LocalMessageStore(BASE_DIR)
    logger.info("Using local message store path=%s", store.describe().get("location"))
    return store


message_store = None

def init_logs_dir():
    global message_store
    if message_store is None:
        message_store = _build_message_store()
    message_store.ensure_ready()


def create_session(session_id: str, tz_local: str):
    if message_store is None:
        init_logs_dir()
    message_store.create_session(session_id, tz_local)


def append_turn_record(session_id: str, record: dict):
    if message_store is None:
        init_logs_dir()
    message_store.append_turn_record(session_id, record)


def update_session_index_row(session_id: str, **kwargs):
    if message_store is None:
        init_logs_dir()
    message_store.update_session_index_row(session_id, **kwargs)


def load_messages_with_logging():
    if message_store is None:
        init_logs_dir()
    try:
        msgs = message_store.load_all_messages()
        location = message_store.describe()
        return msgs, None, location
    except MessageStoreError as exc:
        logger.exception("Messages load failed")
        return [], str(exc), None

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
    init_logs_dir()
    store_info = message_store.describe()
    error = None
    try:
        sessions = message_store.list_sessions()
    except MessageStoreError as exc:
        logger.exception("session list failed")
        sessions = []
        error = str(exc)
    session_detail_template = url_for("serve_session_json", session_id="__SESSION_ID__")
    return render_template(
        "sessions.html",
        store_info=store_info,
        sessions=sessions,
        error=error,
        home_url="/",
        sessions_api=url_for("list_sessions_json"),
        session_detail_template=session_detail_template,
        messages_api=url_for("serve_messages_json"),
    )

@app.get("/logs/files")
def list_sessions_json():
    init_logs_dir()
    try:
        sessions = message_store.list_sessions()
        return jsonify({"sessions": sessions, "store": message_store.describe()})
    except MessageStoreError as exc:
        logger.error("session list failed: %s", exc)
        return jsonify({"error": str(exc)}), 500

@app.get("/logs/sessions/<path:session_id>")
def serve_session_json(session_id):
    init_logs_dir()
    try:
        events = message_store.get_session_messages(session_id)
    except MessageStoreError as exc:
        logger.error("session load failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    if not events:
        return jsonify({"session_id": session_id, "events": []}), 404
    return jsonify({"session_id": session_id, "events": events})

@app.get("/logs/messages")
@app.get("/messages")
def serve_messages_json():
    msgs, err, store_info = load_messages_with_logging()
    if err:
        logger.error("message store load failed: %s", err)
        status = 404 if "not found" in err.lower() else 500
        payload = {"error": err}
        if store_info:
            payload["store"] = store_info
        return jsonify(payload), status
    return jsonify({"messages": msgs, "store": store_info})

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

