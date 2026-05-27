"""QLoRA fine-tuning entry points for NuExtract3."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QLoRAConfig:
    """Hyperparameters for QLoRA fine-tuning."""

    model_id: str = "numind/NuExtract3"
    output_dir: str = "checkpoints/docpii-qlora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 8192
    bf16: bool = True


def train_qlora(
    train_jsonl: str | Path,
    *,
    config: QLoRAConfig | None = None,
) -> Path:
    """Fine-tune NuExtract3 with QLoRA on labeled extraction examples.

    This is a thin wrapper around Hugging Face TRL ``SFTTrainer``. Install
    training dependencies with ``pip install 'docpii[train]'`` before calling.

    Args:
        train_jsonl: JSONL file of labeled documents (see ``docpii.data``).
        config: Optional training hyperparameters.

    Returns:
        Path to the saved adapter checkpoint directory.

    Raises:
        ImportError: If training dependencies are not installed.
    """
    config = config or QLoRAConfig()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Training requires optional dependencies. Install with: pip install 'docpii[train]'"
        ) from exc

    from docpii.data.formats import build_training_messages, load_training_jsonl
    from docpii.templates.nuextract import default_extraction_template

    examples = load_training_jsonl(train_jsonl)
    template = default_extraction_template()
    records = [
        {"messages": build_training_messages(example, template), "id": example.example_id}
        for example in examples
    ]
    dataset = Dataset.from_list(records)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=config.max_seq_length,
        remove_unused_columns=False,
    )

    def formatting_func(batch: dict[str, Any]) -> list[str]:
        """Format a batch row for SFT (placeholder — customize for multimodal collator)."""
        return [msg["content"] if isinstance(msg["content"], str) else str(msg["content"]) for msg in batch["messages"] if msg["role"] == "assistant"]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    logger.info("Starting QLoRA training on %d examples", len(examples))
    trainer.train()
    output_path = Path(config.output_dir)
    trainer.save_model(str(output_path))
    return output_path
