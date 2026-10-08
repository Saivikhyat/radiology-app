import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
BASE_GGUF = MODELS_DIR / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
MMPROJ_GGUF = MODELS_DIR / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
LORA_GGUF = MODELS_DIR / "radiology-lora.gguf"

PORT = int(os.environ.get("LLAMA_PORT", "8081"))
BASE_URL = f"http://127.0.0.1:{PORT}"
SERVER_LOG = Path("/tmp/llama_server.log")

_server_proc: subprocess.Popen | None = None


class LocalModelError(RuntimeError):
    pass


def find_binary() -> str | None:
    local_build = Path(__file__).resolve().parent / ".llama-bin" / "llama-server"
    if local_build.exists():
        return str(local_build)
    for name in ("llama-server", "llama-server-bin"):
        path = shutil.which(name)
        if path:
            return path
    for candidate in ("/usr/local/bin/llama-server", "/opt/homebrew/bin/llama-server"):
        if Path(candidate).exists():
            return candidate
    brew = shutil.which("brew")
    if brew:
        out = subprocess.run([brew, "--prefix", "llama.cpp"], capture_output=True, text=True)
        if out.returncode == 0:
            candidate = Path(out.stdout.strip()) / "bin" / "llama-server"
            if candidate.exists():
                return str(candidate)
    return None


def missing_models() -> list[str]:
    return [str(p) for p in (BASE_GGUF, MMPROJ_GGUF, LORA_GGUF) if not p.exists()]


def _health_ok() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def ensure_server() -> str:
    """Start llama-server once; return its base URL."""
    global _server_proc

    if _health_ok():
        return BASE_URL

    missing = missing_models()
    if missing:
        raise LocalModelError(
            "Missing model files:\n" + "\n".join(missing) + "\nSee README for the download commands."
        )

    binary = find_binary()
    if not binary:
        raise LocalModelError(
            "llama-server not found. Install it with: brew install llama.cpp"
        )

    if _server_proc is not None and _server_proc.poll() is not None:
        _server_proc = None  # crashed earlier

    if _server_proc is None:
        cmd = [
            binary,
            "--model", str(BASE_GGUF),
            "--mmproj", str(MMPROJ_GGUF),
            "--port", str(PORT),
            "--ctx-size", "8192",
        ]
        if LORA_GGUF.exists():
            cmd += ["--lora", str(LORA_GGUF)]
        log = open(SERVER_LOG, "w")
        _server_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    deadline = time.time() + 120
    while time.time() < deadline:
        if _server_proc.poll() is not None:
            tail = SERVER_LOG.read_text()[-800:] if SERVER_LOG.exists() else ""
            _server_proc = None
            raise LocalModelError(f"llama-server exited early. Log tail:\n{tail}")
        if _health_ok():
            return BASE_URL
        time.sleep(1)

    raise LocalModelError(f"llama-server did not become healthy in 120s. Log: {SERVER_LOG}")


def generate(messages: list, timeout: int = 600) -> str:
    """OpenAI-compatible chat completion against the local llama-server."""
    url = ensure_server() + "/v1/chat/completions"
    payload = json.dumps(
        {
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1600,
            "repetition_penalty": 1.2,
            "frequency_penalty": 0.5,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        raise LocalModelError(f"llama-server error {exc.code}: {body}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise LocalModelError(f"Cannot reach local llama-server: {exc}") from exc
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LocalModelError(f"Unexpected response: {json.dumps(data)[:400]}") from exc
