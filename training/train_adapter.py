#!/usr/bin/env python3
"""
Codette LoRA Adapter Training Script
=====================================

Production-ready LoRA fine-tuning for Llama 3.1 8B Instruct.
Supports 4-bit quantization, configurable LoRA parameters,
gradient accumulation, and both CPU and GPU training.

Usage:
    python -m training.train_adapter \
        --dataset datasets/newton_reasoning.jsonl \
        --adapter-name newton \
        --output-dir adapters/newton \
        --epochs 3 \
        --batch-size 2 \
        --lr 2e-4 \
        --lora-rank 16 \
        --max-seq-length 2048
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import Dataset


def setup_logging(output_dir: str, adapter_name: str) -> logging.Logger:
    """Configure logging to both console and file.

    Args:
        output_dir: Directory to write log files.
        adapter_name: Name of the adapter being trained.

    Returns:
        Configured logger instance.
    """
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{adapter_name}_{timestamp}.log"

    logger = logging.getLogger(f"codette.train.{adapter_name}")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler (debug level)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console handler (info level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    return logger


def detect_device() -> tuple[str, bool]:
    """Detect the best available compute device.

    Returns:
        Tuple of (device_string, use_quantization).
        Quantization is only enabled on CUDA GPUs.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return "cuda", True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", False
    else:
        return "cpu", False


def load_training_config(config_path: str | None = None) -> dict:
    """Load the default training configuration from YAML.

    Args:
        config_path: Optional path to a custom config file.
            Falls back to training/configs/default_training.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default_training.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl_dataset(dataset_path: str) -> Dataset:
    """Load a chat-format JSONL dataset.

    Each line should be a JSON object with a 'messages' key containing
    a list of message dicts with 'role' and 'content' keys.

    Args:
        dataset_path: Path to the .jsonl file.

    Returns:
        A HuggingFace Dataset object.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or malformed.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records = []
    malformed_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue

            # Validate structure
            if "messages" not in record:
                malformed_count += 1
                continue

            messages = record["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                malformed_count += 1
                continue

            # Validate each message has role and content
            valid = True
            for msg in messages:
                if not isinstance(msg, dict):
                    valid = False
                    break
                if "role" not in msg or "content" not in msg:
                    valid = False
                    break
                if msg["role"] not in ("system", "user", "assistant"):
                    valid = False
                    break
            if not valid:
                malformed_count += 1
                continue

            records.append(record)

    if not records:
        raise ValueError(
            f"No valid records found in {dataset_path}. "
            f"{malformed_count} malformed lines skipped."
        )

    return Dataset.from_list(records)


def format_chat_messages(example: dict, tokenizer) -> dict:
    """Format a single example using the tokenizer's chat template.

    Applies the chat template to the messages list and tokenizes the result.

    Args:
        example: Dictionary with a 'messages' key.
        tokenizer: A HuggingFace tokenizer with chat template support.

    Returns:
        Dictionary with 'text' key containing the formatted string.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def create_model_and_tokenizer(
    model_name: str,
    use_quantization: bool,
    device: str,
    logger: logging.Logger,
):
    """Load the base model and tokenizer with optional quantization.

    Args:
        model_name: HuggingFace model identifier.
        use_quantization: Whether to apply 4-bit quantization.
        device: Target device string.
        logger: Logger instance.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "use_cache": False,
    }

    if use_quantization and device == "cuda":
        logger.info("Applying 4-bit quantization (NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif device == "mps":
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32
        logger.warning(
            "Training on CPU. This will be extremely slow. "
            "GPU training is strongly recommended."
        )

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device == "mps":
        model = model.to(device)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model loaded: {total_params:,} total params, "
        f"{trainable_params:,} trainable (before LoRA)"
    )

    return model, tokenizer


def apply_lora_config(
    model,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    logger: logging.Logger,
):
    """Apply LoRA configuration to the model.

    Args:
        model: The base model.
        lora_rank: LoRA decomposition rank.
        lora_alpha: LoRA scaling alpha.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        logger: Logger instance.

    Returns:
        Model with LoRA adapters applied.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    logger.info(
        f"LoRA applied: rank={lora_rank}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}"
    )
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({pct:.2f}%)"
    )
    logger.info(f"Target modules: {target_modules}")

    return model


def train(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_seq_length: int,
    warmup_ratio: float,
    logging_steps: int,
    save_steps: int,
    logger: logging.Logger,
):
    """Execute the training loop using SFTTrainer from TRL.

    Args:
        model: The LoRA-adapted model.
        tokenizer: The tokenizer.
        dataset: Formatted training dataset.
        output_dir: Directory for checkpoints and final model.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Steps to accumulate gradients.
        learning_rate: Learning rate for AdamW optimizer.
        max_seq_length: Maximum sequence length for tokenization.
        warmup_ratio: Fraction of steps for LR warmup.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        logger: Logger instance.

    Returns:
        Training result object from Trainer.
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    device = next(model.parameters()).device
    use_fp16 = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    effective_batch = batch_size * gradient_accumulation_steps
    total_steps = (len(dataset) // effective_batch) * epochs

    logger.info(f"Starting training:")
    logger.info(f"  Dataset size: {len(dataset)} examples")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {effective_batch}")
    logger.info(f"  Estimated total steps: {total_steps}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Max seq length: {max_seq_length}")
    logger.info(f"  Output dir: {output_dir}")

    start_time = time.time()
    result = trainer.train()
    elapsed = time.time() - start_time

    logger.info(f"Training complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Final loss: {result.training_loss:.4f}")
    logger.info(f"Total steps: {result.global_step}")

    # Save the final adapter
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final adapter saved to: {final_dir}")

    # Save training metrics
    metrics = {
        "adapter_name": os.path.basename(output_dir),
        "final_loss": result.training_loss,
        "total_steps": result.global_step,
        "training_time_seconds": elapsed,
        "dataset_size": len(dataset),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "timestamp": datetime.now().isoformat(),
    }
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to: {metrics_path}")

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter for Codette on Llama 3.1 8B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL training dataset",
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        required=True,
        help="Name for this adapter (used in logging and output)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for adapter (default: adapters/<adapter-name>)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model name (default: from config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size (default: from config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank (default: from config)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: from config)",
    )
    return parser.parse_args()


def main():
    """Main entry point for adapter training."""
    args = parse_args()

    # Load configuration
    config = load_training_config(args.config)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    # Apply CLI overrides
    model_name = args.model or model_cfg["name"]
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    learning_rate = args.lr or train_cfg["learning_rate"]
    lora_rank = args.lora_rank or lora_cfg["rank"]
    max_seq_length = args.max_seq_length or train_cfg["max_seq_length"]
    output_dir = args.output_dir or os.path.join(
        output_cfg["base_dir"], args.adapter_name
    )

    # Setup logging
    logger = setup_logging(output_dir, args.adapter_name)
    logger.info(f"=== Codette Adapter Training: {args.adapter_name} ===")

    # Detect device
    device, use_quantization = detect_device()
    logger.info(f"Device: {device}, Quantization: {use_quantization}")

    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")

    try:
        # Load dataset
        logger.info(f"Loading dataset: {args.dataset}")
        raw_dataset = load_jsonl_dataset(args.dataset)
        logger.info(f"Loaded {len(raw_dataset)} valid examples")

        # Load model and tokenizer
        model, tokenizer = create_model_and_tokenizer(
            model_name, use_quantization, device, logger
        )

        # Format dataset with chat template
        logger.info("Formatting dataset with chat template")
        formatted_dataset = raw_dataset.map(
            lambda ex: format_chat_messages(ex, tokenizer),
            remove_columns=raw_dataset.column_names,
            desc="Formatting",
        )

        # Apply LoRA
        model = apply_lora_config(
            model=model,
            lora_rank=lora_rank,
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            logger=logger,
        )

        # Train
        result = train(
            model=model,
            tokenizer=tokenizer,
            dataset=formatted_dataset,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            warmup_ratio=train_cfg["warmup_ratio"],
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            logger=logger,
        )

        logger.info(
            f"=== Training complete: {args.adapter_name} "
            f"(loss={result.training_loss:.4f}) ==="
        )

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
