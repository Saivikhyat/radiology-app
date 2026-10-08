# 🩻 Radiology Report Generator

A Streamlit web app that analyzes an uploaded X-ray / MRI image and produces a
structured radiology report (examination, technique, findings, impression,
recommendations) — **entirely offline, for free**, using the
[`SaiVikhyat/qwen3-radiology-lora`](https://huggingface.co/SaiVikhyat/qwen3-radiology-lora)
adapter served by [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Features

- Upload an X-ray / MRI scan (PNG or JPG) and preview it in the browser
- Structured report in five fields: examination, technique, findings,
  impression, recommendations
- Plain, simple language (about an 8th-grade reading level) while keeping
  every finding — medical terms get a short explanation in parentheses
- **Local mode (default):** runs fully offline on your machine, free, no
  token or account needed
- **HF API mode (optional):** same report through Hugging Face Inference
  Providers with your `HF_TOKEN`
- Auto-starts and reuses the `llama-server` backend — no manual server
  management
- One-click download of the report as JSON
- Robust output handling: repetition guards and truncated-JSON salvage

## The model

- **Fine-tuned model:** [`SaiVikhyat/qwen3-radiology-lora`](https://huggingface.co/SaiVikhyat/qwen3-radiology-lora)
- **Base model:** [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
  (loaded as [`unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`](https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit))
- **Adapter:** LoRA rank 16, alpha 16, trained with Unsloth + TRL
- **License:** Apache-2.0

### How it was fine-tuned (Google Colab)

The model was fine-tuned by [SaiVikhyat](https://huggingface.co/SaiVikhyat)
in a Google Colab notebook:

1. **Data** — a radiology report dataset with paired scans and report text,
   downloaded directly from the Hugging Face Hub into the notebook.
2. **Base model** — `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`
   (4-bit quantized Qwen3-VL-4B-Instruct) loaded with Unsloth, which
   accelerates training ~2x and cuts memory so it fits in a Colab GPU.
3. **Training** — supervised fine-tuning with TRL/PEFT using LoRA
   (`r=16`, `alpha=16`), teaching the model to emit radiology report text
   from an input scan.
4. **Upload** — the resulting adapter was pushed to the Hub as
   `SaiVikhyat/qwen3-radiology-lora`, where this app loads it from.

## Inference modes

| Mode | Model | Cost | Needs token |
| --- | --- | --- | --- |
| **Local (default)** | Qwen3-VL-4B-Instruct Q4_K_M + radiology LoRA, run by `llama-server` on `localhost:8081` | Free | No |
| HF API (optional) | `Qwen/Qwen3-VL-4B-Instruct` via Hugging Face Inference Providers | Pay-per-report (fractions of a cent) | `HF_TOKEN` + credits |

Why two modes? The LoRA has **no Inference Provider deployment** (it is an
adapter, not a servable model), so the HF API can only serve its base model.
Local mode runs the real adapter and costs nothing.

## Quick start (this machine — already set up)

```bash
source .venv/bin/activate
streamlit run app.py
```

Open http://localhost:8501, upload a PNG/JPG scan, click **Generate Report**.
First report takes ~1 min (starts the server and loads 2.5 GB); later reports
take ~10–60 s.

## Fresh-machine setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup_local.sh     # ~3.3 GB downloads + one-time build (~15 min)
streamlit run app.py
```

`scripts/setup_local.sh` does everything:

1. Downloads `Qwen3VL-4B-Instruct-Q4_K_M.gguf` + `mmproj` (vision encoder)
2. Clones and builds llama.cpp (cmake/ninja via pip; on Intel Macs builds
   CPU-only — Accelerate/BLAS/Metal headers are broken against current CLT SDKs)
3. Converts `SaiVikhyat/qwen3-radiology-lora` (PEFT, bitsandbytes-4bit base) to
   `models/radiology-lora.gguf` using `requirements-convert.txt` (Python 3.12)

## Optional: HF API mode

1. Buy credits: <https://huggingface.co/settings/inference-providers/billing>
2. `export HF_TOKEN=hf_...` (a token with *Make calls to Inference Providers*)
3. In the sidebar choose **HF API**, generate a report

Override the served model with `HF_MODEL_ID=...` if you deploy the LoRA to an
Inference Endpoint yourself.

## Files

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, report parsing, HF API client |
| `local_llama.py` | Starts/reuses `llama-server`, sends OpenAI-style requests |
| `scripts/setup_local.sh` | One-time local setup (downloads + build + conversion) |
| `requirements-convert.txt` | One-time LoRA→GGUF conversion deps |
| `models/` | GGUF weights + converted adapter (git-ignored, ~3.4 GB) |
| `.llama-bin/` | Built llama.cpp binaries (git-ignored) |

## Troubleshooting

| Error | Cause | Fix |
| --- | --- | --- |
| `command not found: streamlit` | venv not active | `source .venv/bin/activate` |
| `llama-server not found` | not built yet | `bash scripts/setup_local.sh` |
| `Model files missing` | downloads not done | `bash scripts/setup_local.sh` |
| `invalid argument: --no-display-prompt` | stale build | re-run `scripts/setup_local.sh` |
| 402 / "no remaining credits" (API mode) | free HF accounts have $0 credits | add credits or use Local mode |
| Slow generation | CPU-only build | expected on Intel Macs; Apple Silicon uses Metal |

## Notes

- Images are downscaled to 1024 px before inference.
- The GGUF base is Q4_K_M, so the LoRA is applied to a differently-quantized
  base than it was trained on (nf4) — output is close but not bit-identical to
  the original transformers/bitsandbytes setup.
- **Educational/demo only. Not for clinical use.** Every output must be
  reviewed by a qualified radiologist.
