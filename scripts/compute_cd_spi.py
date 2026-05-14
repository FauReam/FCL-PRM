#!/usr/bin/env python3
"""M5: Compute Cross-Domain Step Polysemy Index (CD-SPI).

Usage:
    python scripts/compute_cd_spi.py --config configs/m5_cd_spi.yaml
"""

import argparse
import json

import torch
from transformers import AutoModel, AutoTokenizer

from fclprm.metrics.cd_spi import compute_cd_spi, compute_cd_spi_batch
from fclprm.models.base_wrapper import StepRewardModel
from fclprm.models.checkpoint import load_checkpoint
from fclprm.utils.config import ExperimentConfig
from fclprm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CD-SPI metric")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    config = ExperimentConfig(args.config)
    set_seed(config.get("experiment.seed", 42))

    device = config.get(
        "hardware.device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    backbone_name = config.get("model.backbone")
    checkpoint_dir = config.get("logging.checkpoint_dir", None)

    try:
        tokenizer = AutoTokenizer.from_pretrained(backbone_name, local_files_only=False)
    except OSError as e:
        print(f"[ERROR] Failed to load tokenizer for '{backbone_name}'.")
        print(f"  {e}")
        print("  Please ensure the model name is correct and you have internet access,")
        print(
            "  or download the model locally and set local_files_only=True in config."
        )
        return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    anchor_steps = config.get(
        "anchor.steps",
        [
            "Let x be the variable we want to solve for.",
            "First, we need to identify the key constraints.",
            "Therefore, the answer must be positive.",
            "We can verify this by substitution.",
        ],
    )

    print("[M5] Computing CD-SPI")
    print(f"  Anchor steps: {len(anchor_steps)}")

    num_clients = config.get("federated.num_clients", 4)
    domains = config.get("data.domains", ["math", "code", "medical", "general"])

    # Extract embeddings from each client's model
    all_embeddings = {}
    for i in range(num_clients):
        backbone = AutoModel.from_pretrained(
            backbone_name,
            dtype=torch.float32,
            local_files_only=False,
        )
        model = StepRewardModel(backbone=backbone)

        # Load checkpoint if available; otherwise use initialized model
        if checkpoint_dir:
            import os
            from pathlib import Path

            ckpt_pattern = f"model_m*_r*_c{i}.pt"
            ckpt_files = list(Path(checkpoint_dir).glob(ckpt_pattern))
            if ckpt_files:
                # Pick the latest round
                ckpt_files.sort(key=lambda p: p.name)
                ckpt_path = str(ckpt_files[-1])
                try:
                    load_checkpoint(ckpt_path, model)
                    print(f"  Client {i}: loaded checkpoint {ckpt_path}")
                except Exception as e:
                    print(
                        f"  Client {i}: checkpoint load failed ({e}), using initialized model"
                    )
            else:
                print(f"  Client {i}: no checkpoint found, using initialized model")

        model.to(device)
        model.eval()

        embeddings = []
        with torch.no_grad():
            for step_text in anchor_steps:
                encoded = tokenizer(
                    step_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.get("data.max_length", 512),
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                emb = model.get_step_embedding(input_ids, attention_mask)
                embeddings.append(emb.squeeze(0).cpu())

        all_embeddings[i] = embeddings
        print(f"  Client {i} ({domains[i]}): extracted {len(embeddings)} embeddings")

    # Compute CD-SPI per step
    print("\n[M5] CD-SPI Results:")
    for idx, step_text in enumerate(anchor_steps):
        client_embs = {cid: all_embeddings[cid][idx] for cid in all_embeddings}
        cspi = compute_cd_spi(step_text, client_embs)
        print(f"  Step {idx + 1}: CD-SPI = {cspi:.4f} | {step_text[:50]}...")

    # Compute batch CD-SPI
    batch_result = compute_cd_spi_batch(anchor_steps, all_embeddings)
    avg_cspi = sum(batch_result.values()) / len(batch_result)
    print(f"\n  Average CD-SPI: {avg_cspi:.4f}")

    # Save results
    output_file = f"{config.get('logging.log_dir', './logs')}/cd_spi_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "anchor_steps": anchor_steps,
                "per_step_cspi": {k: float(v) for k, v in batch_result.items()},
                "average_cspi": float(avg_cspi),
            },
            f,
            indent=2,
        )
    print(f"\n[M5] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
