
import gradio as gr
from unsloth import FastLanguageModel
import torch

# Configuration
model_name = "unsloth/llama-3-8b-bnb-4bit"
max_seq_length = 2048

# Load Model and Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Medical Prompt Template
medical_prompt = """Below is a medical question. Write a clinical and accurate answer.

### Question:
{}

### Answer:
{}"""

def generate_response(question, temperature, top_p):
    inputs = tokenizer(
        [medical_prompt.format(question, "")], 
        return_tensors = "pt"
    ).to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 256,
        temperature = temperature,
        top_p = top_p,
        use_cache = True
    )
    response = tokenizer.batch_decode(outputs)[0]
    return response.split("### Answer:")[1].replace("<|end_of_text|>", "").strip()

# Launch Gradio
description = "Clinical triage assistant for HealthDrive. Provides accurate clinical logic for initial patient inquiries."
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Patient Query", placeholder="Describe symptoms..."),
        gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, label="Top-p")
    ],
    outputs=gr.Textbox(label="Clinical Response"),
    title="HealthDrive Triage Assistant",
    description=description
)

if __name__ == "__main__":
    demo.launch()
