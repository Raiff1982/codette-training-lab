#!/usr/bin/env python3
"""
Codette Batch Adapter Training
================================
Train all adapters sequentially from configs/adapter_registry.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

def setup_logging() -> logging.Logger:

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
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ------------------------------------------------------------
# CONFIG LOADING
# ------------------------------------------------------------

def load_adapter_registry(config_path: str) -> dict:

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Adapter registry not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "adapters" not in config:
        raise ValueError("Malformed adapter registry")

    return config["adapters"]


def load_training_config(config_path=None):

    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default_training.yaml"

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------
# OBSERVATORY LOGGING
# ------------------------------------------------------------

def log_metrics_to_observatory(adapter_name, metrics, logger):

    observatory_file = Path("observatory_metrics.json")

    existing = []

    if observatory_file.exists():
        try:
            existing = json.load(open(observatory_file))
        except Exception:
            logger.warning("Observatory file corrupted. Resetting.")

    entry = {
        "type": "adapter_training",
        "adapter": adapter_name,
        "timestamp": datetime.now().isoformat(),
        **metrics
    }

    existing.append(entry)

    with open(observatory_file, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Metrics logged for {adapter_name}")


# ------------------------------------------------------------
# TRAIN SINGLE ADAPTER
# ------------------------------------------------------------

def train_single_adapter(
        adapter_name,
        adapter_config,
        training_defaults,
        output_base_dir,
        logger):

    from training.train_adapter import (
        load_jsonl_dataset,
        create_model_and_tokenizer,
        format_chat_messages,
        apply_lora_config,
        boost_for_training,
        train,
        detect_device
    )

    dataset_path = adapter_config["dataset"]
    output_dir = os.path.join(output_base_dir, adapter_name)

    model_cfg = training_defaults["model"]
    lora_cfg = dict(training_defaults["lora"])      # copy — overrides must not mutate
    train_cfg = dict(training_defaults["training"])

    # Apply per-adapter overrides
    overrides = adapter_config.get("training_overrides", {})
    if "epochs" in overrides:
        train_cfg["epochs"] = overrides["epochs"]
    if "batch_size" in overrides:
        train_cfg["batch_size"] = overrides["batch_size"]
    if "learning_rate" in overrides:
        train_cfg["learning_rate"] = overrides["learning_rate"]
    if "lora_rank" in overrides:
        lora_cfg["rank"] = overrides["lora_rank"]
    if "max_seq_length" in overrides:
        train_cfg["max_seq_length"] = overrides["max_seq_length"]

    logger.info(f"--- Training adapter: {adapter_name} ---")

    start_time = time.time()

    try:

        # DATASET CHECK
        if not Path(dataset_path).exists():
            raise RuntimeError(f"Dataset missing: {dataset_path}")

        raw_dataset = load_jsonl_dataset(dataset_path)

        if len(raw_dataset) == 0:
            raise RuntimeError("Dataset empty")

        logger.info(f"Dataset size: {len(raw_dataset)}")

        # DEVICE
        device = detect_device()

        logger.info(f"Device: {device}")

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            used = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU VRAM total: {vram:.2f} GB")
            logger.info(f"GPU VRAM used: {used:.2f} GB")

        # MODEL LOAD
        logger.info("Loading model")

        model, tokenizer = create_model_and_tokenizer(
            model_cfg["name"], device, logger
        )

        logger.info("Model loaded")

        # DATASET FORMAT
        # XPU: model consumes ~16GB, no headroom for multiprocessing workers
        if device == "xpu":
            cpu_workers = 1
        else:
            cpu_workers = max(1, os.cpu_count() - 1)
        logger.info(f"Formatting dataset with {cpu_workers} workers")

        formatted_dataset = raw_dataset.map(
            lambda ex: format_chat_messages(ex, tokenizer),
            remove_columns=raw_dataset.column_names,
            num_proc=cpu_workers,
            desc=f"Tokenizing {adapter_name}",
        )

        logger.info("Formatting complete")

        # APPLY LORA
        logger.info("Applying LoRA")

        model = apply_lora_config(model, lora_cfg, logger)

        logger.info("LoRA applied")

        # BOOST + TRAIN
        boost_for_training(logger)
        logger.info("Starting training")

        result = train(model, tokenizer, formatted_dataset, train_cfg, output_dir, logger)

        elapsed = time.time() - start_time

        logger.info(
            f"{adapter_name} complete "
            f"(loss={result.training_loss:.4f}, {elapsed:.1f}s)"
        )

        return {
            "status": "success",
            "final_loss": result.training_loss,
            "total_steps": result.global_step,
            "training_time_seconds": elapsed
        }

    except Exception as e:

        elapsed = time.time() - start_time

        logger.error(f"TRAINING FAILURE for {adapter_name}")
        logger.error(str(e))
        logger.error(traceback.format_exc())

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**3
            logger.error(f"GPU memory at crash: {used:.2f} GB")

        return {
            "status": "error",
            "error": str(e),
            "training_time_seconds": elapsed
        }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/adapter_registry.yaml"
    )

    parser.add_argument(
        "--training-config",
        default=None
    )

    parser.add_argument(
        "--adapters",
        nargs="+",
        default=None
    )

    parser.add_argument(
        "--output-dir",
        default="adapters"
    )

    parser.add_argument(
        "--skip-observatory",
        action="store_true"
    )

    args = parser.parse_args()

    logger = setup_logging()

    logger.info("=== Codette Batch Adapter Training ===")

    try:

        registry = load_adapter_registry(args.config)
        training_defaults = load_training_config(args.training_config)

    except Exception as e:

        logger.error("Configuration failure")
        logger.error(str(e))
        sys.exit(1)

    if args.adapters:
        adapter_names = args.adapters
    else:
        adapter_names = list(registry.keys())

    logger.info(f"Adapters: {adapter_names}")

    results = {}

    total_start = time.time()

    for i, adapter_name in enumerate(adapter_names, 1):

        logger.info(f"\n[{i}/{len(adapter_names)}] Training {adapter_name}")

        metrics = train_single_adapter(
            adapter_name,
            registry[adapter_name],
            training_defaults,
            args.output_dir,
            logger
        )

        results[adapter_name] = metrics

        # Aggressive memory cleanup between adapters — model is ~16GB
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetProcessWorkingSetSize(k32.GetCurrentProcess(), -1, -1)
        except Exception:
            pass

        if not args.skip_observatory:
            log_metrics_to_observatory(adapter_name, metrics, logger)

    total_elapsed = time.time() - total_start

    logger.info(f"Training finished in {total_elapsed:.1f}s")

    Path("logs").mkdir(exist_ok=True)

    with open("logs/batch_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if any(r["status"] == "error" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()