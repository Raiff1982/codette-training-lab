#!/usr/bin/env python3
"""
Codette Batch Adapter Training
================================

Train all adapters defined in the adapter registry sequentially.
Reads adapter configurations from configs/adapter_registry.yaml,
trains each one with its specified parameters, and logs metrics
to the observatory.

Usage:
    python -m training.train_all_adapters
    python -m training.train_all_adapters --adapters newton davinci quantum
    python -m training.train_all_adapters --config configs/adapter_registry.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


def setup_logging() -> logging.Logger:
    """Configure the batch training logger.

    Returns:
        Configured logger instance.
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_training_{timestamp}.log"

    logger = logging.getLogger("codette.train_all")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    return logger


def load_adapter_registry(config_path: str) -> dict:
    """Load the adapter registry from YAML.

    Args:
        config_path: Path to the adapter_registry.yaml file.

    Returns:
        Dictionary mapping adapter names to their configurations.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter registry not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "adapters" not in config:
        raise ValueError(f"Malformed adapter registry: missing 'adapters' key")

    return config["adapters"]


def load_training_config(config_path: str | None = None) -> dict:
    """Load the default training configuration.

    Args:
        config_path: Optional path to custom training config.

    Returns:
        Parsed training configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default_training.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_metrics_to_observatory(
    adapter_name: str,
    metrics: dict,
    logger: logging.Logger,
) -> None:
    """Log training metrics to the observatory metrics file.

    Appends training results to a central metrics JSON file for
    tracking across all adapter training runs.

    Args:
        adapter_name: Name of the trained adapter.
        metrics: Dictionary of training metrics.
        logger: Logger instance.
    """
    observatory_file = Path("observatory_metrics.json")

    existing_metrics = []
    if observatory_file.exists():
        try:
            with open(observatory_file, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
                if not isinstance(existing_metrics, list):
                    existing_metrics = [existing_metrics]
        except (json.JSONDecodeError, IOError):
            logger.warning(
                f"Could not read existing observatory metrics; starting fresh"
            )
            existing_metrics = []

    entry = {
        "type": "adapter_training",
        "adapter": adapter_name,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }
    existing_metrics.append(entry)

    with open(observatory_file, "w", encoding="utf-8") as f:
        json.dump(existing_metrics, f, indent=2)

    logger.info(f"Metrics logged to observatory for adapter: {adapter_name}")


def train_single_adapter(
    adapter_name: str,
    adapter_config: dict,
    training_defaults: dict,
    output_base_dir: str,
    logger: logging.Logger,
) -> dict:
    """Train a single adapter using the train_adapter module.

    This function imports and calls the core training functions
    from train_adapter.py to execute the actual training loop.

    Args:
        adapter_name: Name of the adapter to train.
        adapter_config: Configuration for this specific adapter.
        training_defaults: Default training parameters from config.
        output_base_dir: Base directory for adapter outputs.
        logger: Logger instance.

    Returns:
        Dictionary of training metrics for this adapter.
    """
    from training.train_adapter import (
        load_jsonl_dataset,
        create_model_and_tokenizer,
        format_chat_messages,
        apply_lora_config,
        train,
        detect_device,
    )

    dataset_path = adapter_config["dataset"]
    output_dir = os.path.join(output_base_dir, adapter_name)

    # Merge overrides with defaults
    model_cfg = training_defaults["model"]
    lora_cfg = training_defaults["lora"]
    train_cfg = training_defaults["training"]
    overrides = adapter_config.get("training_overrides", {})

    epochs = overrides.get("epochs", train_cfg["epochs"])
    batch_size = overrides.get("batch_size", train_cfg["batch_size"])
    learning_rate = overrides.get("learning_rate", train_cfg["learning_rate"])
    lora_rank = overrides.get("lora_rank", lora_cfg["rank"])
    max_seq_length = overrides.get("max_seq_length", train_cfg["max_seq_length"])

    logger.info(f"--- Training adapter: {adapter_name} ---")
    logger.info(f"  Description: {adapter_config.get('description', 'N/A')}")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Epochs: {epochs}, LR: {learning_rate}, Rank: {lora_rank}")

    # Check dataset exists
    if not Path(dataset_path).exists():
        logger.error(f"  Dataset not found: {dataset_path}")
        return {
            "status": "error",
            "error": f"Dataset not found: {dataset_path}",
            "final_loss": None,
            "total_steps": 0,
            "training_time_seconds": 0,
        }

    start_time = time.time()

    try:
        # Load dataset
        raw_dataset = load_jsonl_dataset(dataset_path)
        logger.info(f"  Loaded {len(raw_dataset)} examples")

        # Detect device
        device, use_quantization = detect_device()
        logger.info(f"  Device: {device}, Quantization: {use_quantization}")

        # Load model and tokenizer
        model, tokenizer = create_model_and_tokenizer(
            model_cfg["name"], use_quantization, device, logger
        )

        # Format dataset
        formatted_dataset = raw_dataset.map(
            lambda ex: format_chat_messages(ex, tokenizer),
            remove_columns=raw_dataset.column_names,
            desc=f"Formatting {adapter_name}",
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

        elapsed = time.time() - start_time
        metrics = {
            "status": "success",
            "final_loss": result.training_loss,
            "total_steps": result.global_step,
            "training_time_seconds": elapsed,
            "dataset_size": len(raw_dataset),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
        }

        logger.info(
            f"  Adapter {adapter_name} trained successfully "
            f"(loss={result.training_loss:.4f}, {elapsed:.1f}s)"
        )

        return metrics

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"  Training failed for {adapter_name}: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "final_loss": None,
            "total_steps": 0,
            "training_time_seconds": elapsed,
        }


def print_summary(results: dict[str, dict], logger: logging.Logger) -> None:
    """Print a training summary table.

    Args:
        results: Dictionary mapping adapter names to their metrics.
        logger: Logger instance.
    """
    logger.info("")
    logger.info("=" * 72)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 72)
    logger.info(
        f"{'Adapter':<20} {'Status':<10} {'Loss':<10} "
        f"{'Steps':<8} {'Time':<12}"
    )
    logger.info("-" * 72)

    total_time = 0.0
    success_count = 0
    error_count = 0

    for name, metrics in results.items():
        status = metrics.get("status", "unknown")
        loss = metrics.get("final_loss")
        steps = metrics.get("total_steps", 0)
        elapsed = metrics.get("training_time_seconds", 0)
        total_time += elapsed

        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        time_str = f"{elapsed:.1f}s"

        if status == "success":
            success_count += 1
        else:
            error_count += 1

        logger.info(
            f"{name:<20} {status:<10} {loss_str:<10} "
            f"{steps:<8} {time_str:<12}"
        )

    logger.info("-" * 72)
    logger.info(
        f"Total: {success_count} succeeded, {error_count} failed, "
        f"{total_time:.1f}s total time"
    )
    logger.info("=" * 72)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train all Codette adapters from the registry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/adapter_registry.yaml",
        help="Path to adapter registry YAML",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Path to training defaults YAML",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="Specific adapter names to train (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="adapters",
        help="Base output directory for trained adapters",
    )
    parser.add_argument(
        "--skip-observatory",
        action="store_true",
        help="Skip logging metrics to observatory",
    )
    return parser.parse_args()


def main():
    """Main entry point for batch adapter training."""
    args = parse_args()
    logger = setup_logging()

    logger.info("=== Codette Batch Adapter Training ===")
    logger.info(f"Registry: {args.config}")

    # Load configurations
    try:
        registry = load_adapter_registry(args.config)
        training_defaults = load_training_config(args.training_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Determine which adapters to train
    if args.adapters:
        adapter_names = args.adapters
        unknown = [n for n in adapter_names if n not in registry]
        if unknown:
            logger.error(
                f"Unknown adapters: {unknown}. "
                f"Available: {list(registry.keys())}"
            )
            sys.exit(1)
    else:
        adapter_names = list(registry.keys())

    logger.info(f"Training {len(adapter_names)} adapters: {adapter_names}")

    # Train each adapter
    results: dict[str, dict] = {}
    total_start = time.time()

    for i, adapter_name in enumerate(adapter_names, 1):
        logger.info(
            f"\n[{i}/{len(adapter_names)}] Training: {adapter_name}"
        )

        adapter_config = registry[adapter_name]
        metrics = train_single_adapter(
            adapter_name=adapter_name,
            adapter_config=adapter_config,
            training_defaults=training_defaults,
            output_base_dir=args.output_dir,
            logger=logger,
        )

        results[adapter_name] = metrics

        # Log to observatory
        if not args.skip_observatory:
            log_metrics_to_observatory(adapter_name, metrics, logger)

    total_elapsed = time.time() - total_start
    logger.info(f"\nAll training completed in {total_elapsed:.1f}s")

    # Print summary
    print_summary(results, logger)

    # Save full results
    results_path = Path("logs") / "batch_training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": total_elapsed,
                "adapters": results,
            },
            f,
            indent=2,
        )
    logger.info(f"Full results saved to: {results_path}")

    # Exit with error if any adapter failed
    if any(m.get("status") == "error" for m in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
