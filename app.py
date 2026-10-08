import base64
import io
import json
import os
import re

import requests
import streamlit as st
from PIL import Image

import local_llama

# The requested LoRA (SaiVikhyat/qwen3-radiology-lora) has no Inference Provider
# deployment, so we call its base model over the API by default. Override with
# HF_MODEL_ID if you deploy the adapter yourself (e.g. an Inference Endpoint).
MODEL_ID = os.environ.get("HF_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
TARGET_LORA = "SaiVikhyat/qwen3-radiology-lora"

CHAT_API_URL = "https://router.huggingface.co/v1/chat/completions"
PROVIDER_API_URL = "https://router.huggingface.co/{provider}/v1/chat/completions"
DEFAULT_PROVIDER = "featherless-ai"
CREDITS_URL = "https://huggingface.co/settings/inference-providers/billing"
MAX_IMAGE_SIDE = 1024

REPORT_KEYS = ("examination", "technique", "findings", "impression", "recommendations")

PROMPT_TEMPLATE = """You are an expert radiologist. Analyze the attached medical scan \
(X-ray, CT, or MRI) and produce a structured radiology report.

Write every field in clear, simple language that a non-medical person can easily \
understand (about an 8th-grade reading level): short sentences and everyday words. \
If a medical term is needed, add a brief plain-language explanation in parentheses \
right after it.

Rules:
- Be thorough and descriptive: 4 to 8 short sentences per field. Walk through
  everything visible on the scan (for example lungs, heart, bones, diaphragm,
  soft tissue) and describe how each part looks.
- Mention each finding exactly once — never repeat or restate a point.
- Do not leave out anything important, and do not soften or change any finding.

Respond ONLY with a JSON object using exactly these keys:
{{
  "examination": "type of study, e.g. Chest X-ray PA",
  "technique": "how the image was acquired/evaluated, in simple words",
  "findings": "everything visible on the scan, in simple words",
  "impression": "plain-language summary of what the scan shows",
  "recommendations": "next steps in plain words, or 'None'"
}}"""


class HF_APIError(RuntimeError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(self._friendly(status, body))

    @staticmethod
    def _friendly(status: int, body: str) -> str:
        snippet = body[:300]
        if status == 402:
            return (
                "Your Hugging Face account has no Inference Providers credits. "
                f"Add pre-paid credits ({CREDITS_URL}) or subscribe to PRO, then retry."
            )
        if status == 404:
            return f"Model not found or not served: {snippet}"
        if status == 401 or status == 403:
            return f"Authentication failed — check HF_TOKEN. {snippet}"
        return f"HF API error {status}: {snippet}"


def get_token() -> str:
    return os.environ.get("HF_TOKEN", "").strip()


def prepare_image(uploaded_file) -> tuple[bytes, str, str]:
    """Return (image_bytes, mime_type, base64_payload), downscaled to MAX_IMAGE_SIDE."""
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    is_png = uploaded_file.type == "image/png" or uploaded_file.name.lower().endswith(".png")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG" if is_png else "JPEG")
    data = buffer.getvalue()
    mime = "image/png" if is_png else "image/jpeg"
    return data, mime, base64.b64encode(data).decode("ascii")


def build_messages(prompt: str, image_b64: str, mime: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                },
            ],
        }
    ]


def _post_chat(url: str, token: str, messages: list, timeout: int = 180) -> str:
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": MODEL_ID,
            "messages": messages,
            "max_tokens": 1600,
            "temperature": 0.2,
            "stream": False,
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise HF_APIError(response.status_code, response.text)
    payload = response.json()
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise HF_APIError(502, json.dumps(payload)[:500]) from exc


def run_generation(token: str, uploaded_file) -> str:
    _, mime, image_b64 = prepare_image(uploaded_file)
    messages = build_messages(PROMPT_TEMPLATE, image_b64, mime)

    try:
        return _post_chat(CHAT_API_URL, token, messages)
    except HF_APIError as exc:
        if exc.status == 402:
            raise  # no credits: retrying other routes is pointless
        # Aggregate router may not route this model — hit the provider directly.
        return _post_chat(PROVIDER_API_URL.format(provider=DEFAULT_PROVIDER), token, messages)


def parse_report(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```", 2)
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        # Salvage fields from a truncated response
        found = dict(re.findall(r'"(\w+)"\s*:\s*"([^"]*)"', cleaned))
        data = {k: found[k] for k in REPORT_KEYS if k in found} or None
    return data if isinstance(data, dict) else None


def render_report(data: dict, raw: str) -> None:
    st.subheader("Structured Radiology Findings")
    for key in REPORT_KEYS:
        if key in data:
            st.markdown(f"**{key.capitalize()}**")
            st.write(str(data[key]))
            st.divider()
    for key, value in data.items():
        if key not in REPORT_KEYS:
            st.markdown(f"**{key.capitalize()}**")
            st.write(str(value))
            st.divider()
    with st.expander("Raw model output"):
        st.text(raw)


def run_local_generation(uploaded_file) -> str:
    _, mime, image_b64 = prepare_image(uploaded_file)
    messages = build_messages(PROMPT_TEMPLATE, image_b64, mime)
    return local_llama.generate(messages)


def main() -> None:
    st.set_page_config(page_title="Radiology Report Generator", page_icon="🩻")
    st.title("🩻 Radiology Report Generator")

    mode = st.sidebar.radio(
        "Inference mode",
        ["Local — free (llama.cpp)", "HF API — paid (Inference Providers)"],
        index=0,
        help="Local runs Qwen3-VL-4B + the radiology LoRA on this machine at no cost. "
        "The HF API uses your account credits.",
    )
    local_mode = mode.startswith("Local")

    if local_mode:
        st.caption(f"Local llama.cpp · Qwen3-VL-4B-Instruct Q4_K_M + `{TARGET_LORA}` LoRA")
        missing = local_llama.missing_models()
        if missing:
            st.error(
                "Model files missing:\n\n" + "\n".join(f"- `{p}`" for p in missing)
                + "\n\nRun the download commands from the README, then reload."
            )
    else:
        st.caption(f"Serving `{MODEL_ID}` via Hugging Face Inference Providers")

    with st.expander("About the model"):
        st.markdown(
            f"- Model: `{TARGET_LORA}` (LoRA on Qwen3-VL-4B-Instruct).\n"
            "- **Local mode** converts the adapter to GGUF and runs it with llama.cpp "
            "on this machine — no API, no credits, no token.\n"
            f"- **HF API mode** serves the base model `{MODEL_ID}` instead (the LoRA has "
            "no Inference Provider deployment) and requires credits."
        )
    st.warning(
        "Educational/demo tool only. Not for clinical use — always have a "
        "qualified radiologist review any finding."
    )

    token = get_token()
    if not local_mode and not token:
        st.error("HF_TOKEN is not set. Export it before launching: `export HF_TOKEN=hf_...`")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an X-ray / MRI image (PNG or JPG)",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), caption=uploaded_file.name, use_container_width=True)

    if st.button("Generate Report", type="primary", disabled=uploaded_file is None):
        spinner = "Starting local model (first run loads ~2.5 GB)..." if local_mode else "Analyzing scan and generating findings..."
        with st.spinner(spinner):
            try:
                if local_mode:
                    raw = run_local_generation(uploaded_file)
                else:
                    raw = run_generation(token, uploaded_file)
            except (HF_APIError, local_llama.LocalModelError) as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")
                st.stop()

        data = parse_report(raw)
        if data:
            render_report(data, raw)
        else:
            st.subheader("Report")
            st.write(raw)
            with st.expander("Raw model output"):
                st.text(raw)

        st.download_button(
            "Download report (JSON)",
            data=json.dumps(data or {"report": raw}, indent=2),
            file_name="radiology_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
