# HealthDrive Clinical Triage Chatbot: Domain-Specific LLM Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ay1uTHwg_Ak1JAbZ71bdPYltOXmyilOF)

## Project Definition & Domain Alignment
I am Ayomide, co-founder of HealthDrive. This project builds a conversational assistant for healthcare triage, providing accurate clinical information to support HealthDrive's mission of improving healthcare access. The chatbot acts as an automated triage engine to handle initial patient questions before they reach human staff.

## Dataset Collection & Preprocessing
I fine-tuned the model on the `medalpaca/medical_meadow_medical_flashcards` dataset from Hugging Face.
* **Cleaning:** Filtered the dataset to remove missing values and empty strings.
* **Tokenization:** Used the Llama-3 BPE tokenizer with a max sequence length of 2048.
* **Normalization:** Formatted data into a strict instruction-response template.

## Model Fine-tuning
I used the `unsloth/llama-3-8b-bnb-4bit` model with Parameter-Efficient Fine-Tuning (PEFT) via LoRA.

### Experiment Table
| Experiment | Learning Rate | Batch Size | Optimizer | Peak GPU Memory | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Run 1 (Baseline) | 2e-5 | 2 | adamw_torch | ~6.4 GB | High loss (1.8+), slow convergence. |
| Run 2 | 5e-5 | 4 | adamw_8bit | OOM Error | Batch size too large for T4 GPU. |
| **Run 3 (Final)** | **2e-4** | **2 (Grad Accum: 4)** | **adamw_8bit** | **~7.2 GB** | **Loss dropped to 0.85. Optimal.** |



## Performance Metrics
* **BLEU Score:** 17.77 (**104% improvement** over baseline 8.71).
* **ROUGE-1:** 0.552 | **ROUGE-L:** 0.486.
* **Qualitative Test:** Successfully identifies the relationship between Mg2+, PTH, and Ca2+ levels, which the base model failed.

## UI Integration
Deployed via a **Gradio** web interface.
* **Features:** Includes advanced inference controls (Temperature, Top-P), clinical safety disclaimers, and interactive patient scenario examples.

## How to Run
1. Open the [Colab Notebook](https://colab.research.google.com/drive/1ay1uTHwg_Ak1JAbZ71bdPYltOXmyilOF).
2. Ensure Runtime is set to **T4 GPU**.
3. Click **Runtime > Run all**.
4. Access the Gradio public link at the bottom of the notebook.
