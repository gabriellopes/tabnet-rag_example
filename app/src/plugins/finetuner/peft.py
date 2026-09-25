# app/src/plugins/finetuner/peft.py
import os
from typing import Any, Dict, Optional
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from app.src.plugins.finetuner.base import BaseFineTuner


class PEFTFineTuner(BaseFineTuner):
    """PEFT (LoRA/QLoRA) Fine-Tuner plugin for efficient local model adaptation."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        output_dir: str = "models/peft_adapter",
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__(model_id=model_id, output_dir=output_dir)
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.model = None
        self.tokenizer = None

    def prepare_dataset(self, dataset_path: str) -> Any:
        """Loads JSONL dataset created by SFT dataset generator."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
        return load_dataset("json", data_files=dataset_path, split="train")

    def _setup_lora(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    def train(
        self,
        dataset_path: str,
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 2e-4,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Runs LoRA fine-tuning loop on the SPARQL dataset."""
        dataset = self.prepare_dataset(dataset_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        peft_config = self._setup_lora()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="messages",
            max_seq_length=1024,
            tokenizer=self.tokenizer,
            args=training_args,
        )

        train_result = trainer.train()
        self.save_adapter()
        return {"loss": train_result.training_loss}

    def save_adapter(self, path: Optional[str] = None) -> None:
        save_path = path or self.output_dir
        if self.model is not None:
            self.model.save_pretrained(save_path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(save_path)
            print(f"[PEFT Plugin] LoRA adapter successfully saved to: {save_path}")