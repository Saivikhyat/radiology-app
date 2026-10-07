# 🩻 Radiology Report Generator

A Streamlit web app that analyzes an uploaded X-ray / MRI image and generates a
structured radiology report using the Hugging Face Inference API with the model
[`SaiVikhyat/qwen3-radiology-lora`](https://huggingface.co/SaiVikhyat/qwen3-radiology-lora).

## Features

- Upload a PNG or JPG X-ray / MRI scan
- Calls the Hugging Face Inference API using `HF_TOKEN` from the environment
- Returns structured findings: examination, technique, findings, impression,
  recommendations
- One-click download of the report as JSON
- Falls back from the OpenAI-compatible chat router to the classic
  Inference API if the vision request is rejected

## Requirements

- Python 3.9+
- A Hugging Face access token with read permissions

## Setup

```bash
git clone <this-repo>
cd radiology-app

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

export HF_TOKEN=hf_your_token_here   # Windows: set HF_TOKEN=hf_your_token_here
```

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Usage

1. Confirm `HF_TOKEN` is set (the app shows an error banner if it is missing).
2. Upload an X-ray or MRI image (PNG/JPG).
3. Click **Generate Report**.
4. Review the structured findings and optionally download the JSON report.

## Notes

- Images are downscaled to 1024 px on the longest side before being sent to the API.
- This tool is for demonstration purposes only and must **not** be used for
  clinical decision-making. All outputs require review by a qualified radiologist.
