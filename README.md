<div align="center">
  
  <img src="https://img.shields.io/badge/Language-Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-Qwen_2.5_1.5B-FF6B35?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tech-LoRA_Fine--Tuning-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Serving-vLLM_0.7.2-00C7B7?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Load_Testing-Locust-5B8C5A?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge"/>
  
  # Arabic Semantic Chunker (LLM-Based)
  
  **A Multilingual Semantic Text Segmentation System Powered by Large Language Models**
  
  *Fine-tuned Qwen 2.5 1.5B model using LoRA for Semantic Chunking of Arabic, English, and Code-Switched texts—featuring an end-to-end pipeline from synthetic data generation to production-grade vLLM serving.*
</div>

---

## Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Usage & API Examples](#-usage--api-examples)
- [Performance & Load Testing](#-performance--load-testing)

---

## Project Overview
The **Arabic Semantic Chunker** represents a complete LLMOps lifecycle. It solves a critical issue in modern Retrieval-Augmented Generation (RAG) systems: traditional text splitters (chunking by character count or punctuation) destroy semantic context, especially in complex languages like Arabic. 

This project leverages a fine-tuned LLM capable of understanding semantic boundaries, ensuring that documents are split into logically coherent, self-contained chunks without losing contextual integrity.

---

## Key Features
| Feature | Description |
|---|---|
| ** Multilingual Support** | Processes Modern Standard Arabic, English, and technical code-switched texts seamlessly. |
| ** Structured Outputs** | Guarantees syntactically valid JSON outputs strictly adhering to a Pydantic schema. |
| ** High-Throughput Serving** | Deployed via vLLM with dynamic LoRA adapter loading for production readiness. |
| ** Quality Assurance** | Automated filtering of anomalies (e.g., hallucinatory characters) post-training. |
| ** Load Tested** | Stress-tested using Locust, simulating high-concurrency environments. |
| ** Resource Efficient** | Trains and serves a 1.5B parameter model effectively on a single 15GB VRAM GPU (T4). |

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    End-to-End LLMOps Pipeline                   │
└─────────────────────────────────────────────────────────────────┘
  Phase 1                 Phase 2                  Phase 3
┌──────────────┐        ┌──────────────┐        ┌──────────────────┐
│ Synthetic    │        │ LoRA         │        │ vLLM Production  │
│ Data Gen     │ ──────►│ Fine-Tuning  │ ──────►│ Serving          │
│              │        │              │        │                  │
│ • Faker      │        │ • Unsloth    │        │ • REST API       │
│ • Gemini     │        │ • Qwen2.5    │        │ • Dynamic LoRA   │
│ • 2700+ rows │        │ • 1000 steps │        │ • Port 8000      │
└──────────────┘        └──────────────┘        └────────┬─────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │ Stress Testing   │
                                              │ (Locust)         │
                                              │                  │
                                              │ • 5-20 Users     │
                                              │ • 60 Seconds     │
                                              │ • Tokens/sec     │
                                              └──────────────────┘
```

### Internal Processing Flow

```text
Raw Text (Arabic/English/Mixed)
        │
        ▼
┌───────────────────┐
│  Chat Template    │  ← System Message + Strict Schema constraints
│  (Qwen Format)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Qwen 2.5 1.5B    │
│  + LoRA Adapter   │  ← Mo-Abdelfattah/arabic-semantic-chunker-qwen1.5b
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  JSON Output      │  ← Post-processing & validation via json_repair
│  (Pydantic)       │
└────────┬──────────┘
         │
         ▼
{
  "original_text_length": 342,
  "semantic_chunks": [
    "First independent semantic chunk...",
    "Second semantic chunk..."
  ]
}
```

---

## Repository Structure

```text
arabic-semantic-chunker/
│
├── notebooks/
│   ├── 01_data_generation.ipynb      # Synthetic data generation via Gemini & Faker
│   ├── 02_fine_tuning.ipynb          # SFT pipeline using Unsloth & LoRA
│   ├── 03_evaluation.ipynb           # Model evaluation and output sanitization
│   └── 04_deployment_load_test.ipynb # vLLM serving and Locust load testing
│
├── scripts/
│   ├── data/
│   │   ├── generate_synthetic.py     
│   │   └── validate_dataset.py       
│   ├── serving/
│   │   ├── start_server.sh           # vLLM initialization script
│   │   └── locustfile.py             # Locust stress testing scenarios
│   └── evaluation/
│       └── run_eval.py               
│
├── configs/
│   ├── model_config.yaml             
│   └── serving_config.yaml           
│
├── requirements.txt               
└── README.md
```

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **GPU VRAM** | 12 GB | 15 GB (e.g., T4, L4, A10g) |
| **RAM** | 12 GB | 16 GB+ |
| **Python** | 3.10 | 3.12 |
| **CUDA** | 12.1 | 12.4+ |
| **OS** | Linux | Ubuntu 22.04 |

---

## Installation

**Step 1: Clone the repository**
```bash
git clone [https://github.com/Mo-Abdelfattah/arabic-semantic-chunker.git](https://github.com/Mo-Abdelfattah/arabic-semantic-chunker.git)
cd arabic-semantic-chunker
```

**Step 2: Install Dependencies**
```bash
# Note: Install numpy<2 first to avoid vLLM compatibility issues
pip install -r requirements.txt
```
*(For Unsloth installation specific to your CUDA version, please refer to the [official Unsloth repository](https://github.com/unslothai/unsloth).)*

**Step 3: Spin up the vLLM Server**
```bash
BASE_MODEL="unsloth/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH="Mo-Abdelfattah/arabic-semantic-chunker-qwen1.5b"

nohup vllm serve "$BASE_MODEL" \
  --dtype=half \
  --gpu-memory-utilization 0.8 \
  --max-model-len 2048 \
  --max-lora-rank 64 \
  --enable-lora \
  --lora-modules arabic-chunker="$ADAPTER_PATH" \
  > nohup.out 2>&1 &
```

---

## Usage & API Examples

### Python REST API Call

```python
import requests
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct")

# Mixed-Language Input Text
input_text = """
أسلوب إديسون في العمل كان يعتمد على Trial and Error.
كان يمتلك فريقًا في مختبره الشهير في Menlo Park.
رؤيته للكهرباء ارتكزت على التيار المستمر Direct Current (DC).
"""

system_message = (
    "You are a professional multilingual NLP data parser.\n"
    "Split the provided text into meaningful, self-contained semantic chunks.\n"
    "Preserve the original text exactly and respect its language."
)

schema_str = '{"properties": {"original_text_length": {"type": "integer"}, "semantic_chunks": {"items": {"type": "string"}, "minItems": 2, "type": "array"}}, "required": ["original_text_length", "semantic_chunks"], "type": "object"}'

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": f"# Input Text:\n{input_text}\n\n# Output Schema:\n{schema_str}\n\n# Output JSON:\n```json\n"}
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

response = requests.post("http://localhost:8000/v1/completions", json={
    "model": "arabic-chunker",
    "prompt": prompt,
    "max_tokens": 1000,
    "temperature": 0.1
})

print(response.json()["choices"][0]["text"])
```

**Output:**
```json
{
  "original_text_length": 198,
  "semantic_chunks": [
    "أسلوب إديسون في العمل كان يعتمد على Trial and Error.",
    "كان يمتلك فريقًا في مختبره الشهير في Menlo Park.",
    "رؤيته للكهرباء ارتكزت على التيار المستمر Direct Current (DC)."
  ]
}
```

---

## Performance & Load Testing

Stress testing was conducted using **Locust** on a Google Colab T4 (15 GB VRAM) instance:

| Metric | Result |
|---|---|
| **Concurrent Users** | 5 (Optimized for T4 VRAM limits) |
| **Test Duration** | 60 Seconds |
| **Total Processed Tokens** | ~37,000 tokens |
| **Throughput** | **~393 tokens/sec** |
| **Failure Rate** | 0.00% |

> **Tuning Note:** For optimal throughput on a single T4 without encountering OOM errors, ensure `--max-model-len 2048` and `--gpu-memory-utilization 0.8` are strictly set.

---

## License
This project is licensed under the Apache License 2.0.
