import os, json, csv, uuid, time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

LOGS_DIR = os.getenv("LOGS_DIR", "logs")
SESSIONS_DIR = os.path.join(LOGS_DIR, "sessions")
MESSAGES_PATH = os.path.join(SESSIONS_DIR, "messages.json")  # JSON Lines (one JSON per line)
INDEX_CSV = os.path.join(LOGS_DIR, "index.csv")

# ---------- utils ----------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _lock_path(path: str) -> str:
    return f"{path}.lock"

def _acquire_lock(path: str, timeout_s: float = 5.0, poll_s: float = 0.05) -> None:
    lock = _lock_path(path)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Lock timeout: {lock}")
            time.sleep(poll_s)

def _release_lock(path: str) -> None:
    lock = _lock_path(path)
    try:
        os.remove(lock)
    except FileNotFoundError:
        pass

def _atomic_write(path: str, data: str) -> None:
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(data)
    os.replace(tmp, path)

# ---------- public API ----------

def init_logs_dir() -> None:
    _ensure_dir(LOGS_DIR)
    _ensure_dir(SESSIONS_DIR)
    # seed messages file
    if not os.path.exists(MESSAGES_PATH):
        with open(MESSAGES_PATH, "w", encoding="utf-8") as f:
            pass  # empty JSONL file
    # seed index
    if not os.path.exists(INDEX_CSV):
        header = ["session_id", "started_ts_utc", "last_ts_utc", "model_name", "turns", "last_decision", "tz_local"]
        with open(INDEX_CSV, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)

def create_session(session_id: str, tz_local: str) -> None:
    """Create or upsert the session row in index.csv."""
    started = _utc_now_iso()
    _acquire_lock(INDEX_CSV)
    try:
        rows = _read_index()
        ids = {r["session_id"] for r in rows}
        if session_id not in ids:
            rows.append({
                "session_id": session_id,
                "started_ts_utc": started,
                "last_ts_utc": started,
                "model_name": "",
                "turns": "0",
                "last_decision": "",
                "tz_local": tz_local,
            })
            _write_index(rows)
    finally:
        _release_lock(INDEX_CSV)

def append_turn_record(session_id: str, record: Dict[str, Any]) -> None:
    """Append one JSON object to messages.json (newline-delimited)."""
    rec = dict(record)
    rec.setdefault("session_id", session_id)
    rec.setdefault("ts_utc", _utc_now_iso())

    line = json.dumps(rec, ensure_ascii=False)
    _acquire_lock(MESSAGES_PATH)
    try:
        with open(MESSAGES_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    finally:
        _release_lock(MESSAGES_PATH)

def update_session_index_row(session_id: str, last_ts_utc: str, model_name: str, decision: Optional[str]) -> None:
    _acquire_lock(INDEX_CSV)
    try:
        rows = _read_index()
        found = False
        for r in rows:
            if r["session_id"] == session_id:
                r["last_ts_utc"] = last_ts_utc or _utc_now_iso()
                if model_name:
                    r["model_name"] = model_name
                try:
                    r["turns"] = str(int(r.get("turns", "0")) + 1)
                except ValueError:
                    r["turns"] = "1"
                if decision:
                    r["last_decision"] = decision
                found = True
                break
        if not found:
            rows.append({
                "session_id": session_id,
                "started_ts_utc": last_ts_utc or _utc_now_iso(),
                "last_ts_utc": last_ts_utc or _utc_now_iso(),
                "model_name": model_name or "",
                "turns": "1",
                "last_decision": decision or "",
                "tz_local": "",
            })
        _write_index(rows)
    finally:
        _release_lock(INDEX_CSV)

# ---------- index helpers ----------

def _read_index() -> List[Dict[str, str]]:
    if not os.path.exists(INDEX_CSV):
        return []
    with open(INDEX_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def _write_index(rows: List[Dict[str, str]]) -> None:
    header = ["session_id", "started_ts_utc", "last_ts_utc", "model_name", "turns", "last_decision", "tz_local"]
    _atomic_write(INDEX_CSV, _rows_to_csv(header, rows))

def _rows_to_csv(header: List[str], rows: List[Dict[str, str]]) -> str:
    from io import StringIO
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in header})
    return buf.getvalue()
