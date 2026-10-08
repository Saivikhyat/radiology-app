#!/usr/bin/env bash
# One-time local setup: downloads models, builds llama.cpp, converts the LoRA.
# Tested on Intel macOS (x86_64) with Python 3.12/3.13. Idempotent - re-run safe.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
LLAMA_SRC="$ROOT/third_party/llama.cpp"
BIN_DIR="$ROOT/.llama-bin"
MODELS="$ROOT/models"

say() { printf '\n==> %s\n' "$*"; }

[ -x "$PY" ] || { say "creating venv"; python3 -m venv .venv; }

say "app dependencies"
"$ROOT/.venv/bin/pip" install -q -r requirements.txt

# ---------------------------------------------------------------- model files
say "downloading model files (~3.3 GB)"
"$ROOT/.venv/bin/pip" install -q -U huggingface_hub
mkdir -p "$MODELS"
if [ ! -f "$MODELS/Qwen3VL-4B-Instruct-Q4_K_M.gguf" ]; then
  "$ROOT/.venv/bin/hf" download Qwen/Qwen3-VL-4B-Instruct-GGUF \
    Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-F16.gguf \
    --local-dir "$MODELS"
fi
if [ ! -d "$MODELS/radiology-lora/adapter_model.safetensors" ]; then
  "$ROOT/.venv/bin/hf" download SaiVikhyat/qwen3-radiology-lora --local-dir "$MODELS/radiology-lora"
fi

# ---------------------------------------------------------------- llama.cpp
say "building llama.cpp (cmake via pip)"
"$ROOT/.venv/bin/pip" install -q cmake ninja
CMAKE="$ROOT/.venv/bin/cmake"

if [ ! -d "$LLAMA_SRC" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
fi

# Intel Macs: Accelerate/BLAS/Metal headers break against current CLT SDKs.
# Apple Silicon keeps Metal enabled.
EXTRA=()
if [ "$(uname -m)" = "x86_64" ]; then
  EXTRA+=(-DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF)
fi

(
  cd "$LLAMA_SRC"
  "$CMAKE" -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF "${EXTRA[@]}"
  "$CMAKE" --build build -j"$(sysctl -n hw.ncpu)" --target llama-server llama-mtmd-cli
)

say "installing binaries to $BIN_DIR"
mkdir -p "$BIN_DIR"
cp -a "$LLAMA_SRC/build/bin/." "$BIN_DIR/"
# Build rpath points into the build dir; rewrite so binaries are relocatable.
for f in "$BIN_DIR"/llama-server "$BIN_DIR"/llama-mtmd-cli "$BIN_DIR"/lib*.dylib; do
  if otool -l "$f" 2>/dev/null | grep -q "$LLAMA_SRC/build/bin"; then
    install_name_tool -delete_rpath "$LLAMA_SRC/build/bin" "$f" 2>/dev/null || true
    install_name_tool -add_rpath @loader_path "$f" 2>/dev/null || true
  fi
done

# ------------------------------------------------------- LoRA -> GGUF convert
if [ ! -f "$MODELS/radiology-lora.gguf" ]; then
  say "converting LoRA adapter to GGUF"
  CONV_PY="$(command -v python3.12 || true)"
  [ -n "$CONV_PY" ] || { echo "python3.12 required for conversion (xcode-select --install)"; exit 1; }
  [ -d "$ROOT/.venv-convert" ] || "$CONV_PY" -m venv "$ROOT/.venv-convert"
  "$ROOT/.venv-convert/bin/pip" install -q -r requirements-convert.txt

  # Base config without bitsandbytes metadata (converter cannot dequantize bnb).
  mkdir -p "$MODELS/.base-cfg"
  "$ROOT/.venv/bin/hf" download unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit \
    config.json tokenizer.json tokenizer_config.json chat_template.jinja \
    generation_config.json --local-dir "$MODELS/.base-cfg"
  "$PY" - "$MODELS/.base-cfg/config.json" <<'PYEOF'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c.pop("quantization_config", None)
json.dump(c, open(p, "w"), indent=2)
PYEOF

  # Converter lives in the llama.cpp repo (imports its `conversion` package).
  curl -sL -o "$LLAMA_SRC/convert_lora_to_gguf.py" \
    https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_lora_to_gguf.py
  (cd "$LLAMA_SRC" && "$ROOT/.venv-convert/bin/python" convert_lora_to_gguf.py \
    --base "$MODELS/.base-cfg" \
    --outfile "$MODELS/radiology-lora.gguf" --outtype f16 \
    "$MODELS/radiology-lora")
fi

say "done - run: streamlit run app.py"
