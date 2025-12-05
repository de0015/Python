# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "datasets",
#     "peft",
#     "trl",
#     "accelerate",
#     "bitsandbytes",
#     "huggingface_hub",
# ]
# ///

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import os

# Configuration
MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"
DATASET_NAME = "Firemedic15/conflict-analysis-combined"
OUTPUT_MODEL = "Firemedic15/ministral-conflict-analyst"

print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)

print(f"Loading model: {MODEL_NAME}")
# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training configuration
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    push_to_hub=True,
    hub_model_id=OUTPUT_MODEL,
    hub_token=os.environ.get("HF_TOKEN"),
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving and pushing to hub...")
trainer.save_model()
trainer.push_to_hub()

print(f"Training complete! Model pushed to: https://huggingface.co/{OUTPUT_MODEL}")
