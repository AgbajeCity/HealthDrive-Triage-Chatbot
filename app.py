import os

import gradio as gr
from unsloth import FastLanguageModel

# The fine-tuned QLoRA adapter published on Hugging Face. Unsloth reads the
# adapter config and loads the 4-bit Llama-3-8B base model underneath it.
ADAPTER_ID = os.getenv("HEALTHDRIVE_ADAPTER", "AgbajeCity/healthdrive-triage-llama3-8b-lora")
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Same template the adapter was trained on. Changing it lowers answer quality.
PROMPT = """Below is a medical question. Write a clinical and accurate answer.

### Question:
{}

### Answer:
"""

DISCLAIMER = (
    "Research prototype for education only. It does not diagnose, prescribe or replace "
    "a clinician. In an emergency, contact your nearest health facility or emergency service."
)


def generate_response(question, temperature, top_p):
    question = (question or "").strip()
    if not question:
        return "Please type a question."
    inputs = tokenizer([PROMPT.format(question)], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=temperature > 0,
        temperature=max(temperature, 0.01),
        top_p=top_p,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the new tokens, then stop if the model starts a new question.
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer.split("### Question")[0].strip()


demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Medical question", placeholder="e.g. What are the thyroid side effects of amiodarone?"),
        gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature (0 = most consistent)"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="HealthDrive Triage Assistant",
    description=f"Llama-3-8B fine-tuned with QLoRA on medical flashcards. {DISCLAIMER}",
    examples=[
        ["What are the thyroid-related side effects of amiodarone?", 0.2, 0.9],
        ["Why do aminoglycosides not work on obligate anaerobic bacteria?", 0.2, 0.9],
    ],
)

if __name__ == "__main__":
    demo.launch()
