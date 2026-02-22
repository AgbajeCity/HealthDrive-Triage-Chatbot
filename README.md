# HealthDrive Clinical Triage Chatbot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ay1uTHwg_Ak1JAbZ71bdPYltOXmyilOF)

## Project Definition & Domain Alignment
I am Ayomide, the co-founder of HealthDrive. This project implements a domain-specific conversational assistant for healthcare triage. The tool supports the HealthDrive mission by providing accurate clinical guidance during initial patient inquiries. Automated triage is essential in resource-limited settings to manage patient flow and ensure critical cases receive priority.

## Repository Structure
```text
HealthDrive-Triage-Chatbot/
├── Notebook/
│   └── HealthDrive_Chatbot - LLM Fine-Tuning.ipynb       # Implementation & Fine-tuning pipeline
├── Data/                                       # Dataset documentation
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation

## 📊 Dataset Collection & Preprocessing
The model was fine-tuned on the `medalpaca/medical_meadow_medical_flashcards` dataset from Hugging Face.



- **Data Cleaning:** I filtered the dataset to remove null values and empty strings. This maintained high data integrity.
- **Tokenization:** I used the Llama-3 BPE-based tokenizer with a 2048 token context window.
- **Normalization:** All data followed a strict instruction-response template: "Below is a medical question. Write a clinical and accurate answer."

## 🛠️ Fine-Tuning Methodology
I used QLoRA (4-bit quantization) via the `peft` and `unsloth` libraries. This allowed for efficient training on the Google Colab T4 GPU.

### Experiment Documentation
| Run | Learning Rate | Batch Size | Optimizer | Peak GPU Memory | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 1** | 2e-5 | 2 | adamw_torch | ~6.4 GB | High loss (1.8+). Convergence was slow. |
| **Run 2** | 5e-5 | 4 | adamw_8bit | OOM Error | Batch size exceeded T4 GPU limits. |
| **Run 3** | **2e-4** | **2 (Accum: 4)** | **adamw_8bit** | **~7.2 GB** | **Optimal. Loss dropped to 0.85.** |



## 📈 Performance Metrics & Analysis
- **Quantitative Improvement:** The fine-tuned model achieved a BLEU score of 17.77. This represents a 104% improvement over the 8.71 baseline.
- **Qualitative Testing:** In comparative tests, the fine-tuned model correctly identified the physiological relationship between Magnesium, PTH, and Calcium levels. The base Llama-3 model failed this specific clinical logic test.

## 🚀 Deployment & UI Integration
The chatbot is deployed via a Gradio web interface optimized for clinical interaction.
- **Clinical Controls:** The interface includes adjustable Temperature and Top-P sliders for response precision.
- **Interactive Examples:** Pre-loaded clinical scenarios like "Symptoms of high fever" guide the user.
- **Safety Profile:** I integrated medical disclaimers to ensure responsible and ethical AI use.

## 🌍 Impact
This project demonstrates the effectiveness of parameter-efficient fine-tuning for specialized healthcare tasks. By automating the first layer of triage, HealthDrive can scale healthcare delivery. This provides immediate support to communities with limited clinical access.
