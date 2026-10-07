import base64
import io
import json
import os

import requests
import streamlit as st
from PIL import Image

MODEL_ID = "SaiVikhyat/qwen3-radiology-lora"
CHAT_API_URL = "https://router.huggingface.co/v1/chat/completions"
INFERENCE_API_URL = "https://api-inference.huggingface.co/models/{model_id}"
MAX_IMAGE_SIDE = 1024

REPORT_KEYS = ("examination", "technique", "findings", "impression", "recommendations")

PROMPT_TEMPLATE = """You are an expert radiologist. Analyze the attached medical scan \
(X-ray, CT, or MRI) and produce a structured radiology report.

Respond ONLY with a JSON object using exactly these keys:
{{
  "examination": "type of study, e.g. Chest X-ray PA",
  "technique": "how the image was acquired/evaluated",
  "findings": "detailed observations of visible anatomy and abnormalities",
  "impression": "concise diagnostic summary and clinical impression",
  "recommendations": "follow-up recommendations, or 'None'"
}}"""


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


def call_chat_api(token: str, messages: list, timeout: int = 120) -> str:
    response = requests.post(
        CHAT_API_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_ID,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.2,
            "stream": False,
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Chat API error {response.status_code}: {response.text[:500]}")
    return response.json()["choices"][0]["message"]["content"]


def call_inference_api(token: str, image_bytes: bytes, timeout: int = 120) -> str:
    """Fallback: classic text-generation Inference API with the image attached."""
    url = INFERENCE_API_URL.format(model_id=MODEL_ID)
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params={
            "parameters": json.dumps(
                {
                    "max_new_tokens": 1024,
                    "temperature": 0.2,
                    "return_full_text": False,
                    "prompt": PROMPT_TEMPLATE,
                }
            )
        },
        data=image_bytes,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Inference API error {response.status_code}: {response.text[:500]}")
    return response.text


def run_generation(token: str, uploaded_file) -> str:
    _, mime, image_b64 = prepare_image(uploaded_file)
    try:
        return call_chat_api(token, build_messages(PROMPT_TEMPLATE, image_b64, mime))
    except Exception as chat_error:
        uploaded_file.seek(0)
        image_bytes, _, _ = prepare_image(uploaded_file)
        try:
            return call_inference_api(token, image_bytes)
        except Exception as inference_error:
            raise RuntimeError(
                f"Vision request failed ({chat_error}); fallback failed too ({inference_error})"
            )


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
        return None
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


def main() -> None:
    st.set_page_config(page_title="Radiology Report Generator", page_icon="🩻")
    st.title("🩻 Radiology Report Generator")
    st.caption(f"Model: `{MODEL_ID}` (Hugging Face Inference API)")
    st.warning(
        "Educational/demo tool only. Not for clinical use — always have a "
        "qualified radiologist review any finding."
    )

    token = get_token()
    if not token:
        st.error("HF_TOKEN is not set. Export it before launching: `export HF_TOKEN=hf_...`")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an X-ray / MRI image (PNG or JPG)",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), caption=uploaded_file.name, use_container_width=True)

    if st.button("Generate Report", type="primary", disabled=uploaded_file is None):
        with st.spinner("Analyzing scan and generating findings..."):
            try:
                raw = run_generation(token, uploaded_file)
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
