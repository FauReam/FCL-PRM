#!/usr/bin/env python3
"""Standalone PRM evaluation script.

Usage:
    python scripts/evaluate_prm.py --model_path checkpoints/model.pt --data_dir ./data/prm800k
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from fclprm.data.prm800k import PRM800KLoader
from fclprm.data.utils import collate_step_batch
from fclprm.data.versa_loader import VersaPRMLoader
from fclprm.models.base_wrapper import StepRewardModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PRM model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="EleutherAI/pythia-1.4b",
        help="Backbone model name",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to evaluation data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="prm800k",
        choices=["prm800k", "versaprm"],
        help="Dataset type",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain for VersaPRM (e.g., math, code)",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Data split (for PRM800K)"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print(f"[EVAL] Loading model from: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.backbone,
            local_files_only=False,
        )
    except OSError as e:
        print(f"[ERROR] Failed to load tokenizer for '{args.backbone}'.")
        print(f"  {e}")
        print("  Please ensure the model name is correct and you have internet access,")
        print(
            "  or download the model locally and set local_files_only=True in config."
        )
        return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        backbone = AutoModel.from_pretrained(
            args.backbone,
            dtype=torch.float32,
            local_files_only=False,
        )
    except OSError as e:
        print(f"[ERROR] Failed to load backbone for '{args.backbone}'.")
        print(f"  {e}")
        print("  Please ensure the model name is correct and you have internet access,")
        print(
            "  or download the model locally and set local_files_only=True in config."
        )
        return
    model = StepRewardModel(backbone=backbone)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    print(f"[EVAL] Loading data from: {args.data_dir}")
    if args.dataset == "versaprm":
        loader = VersaPRMLoader(data_dir=args.data_dir)
        if args.domain:
            raw_samples = loader.load_domain(args.domain)
            print(f"  Domain: {args.domain}, {len(raw_samples)} samples")
        else:
            raw_samples = loader.load()
            print(f"  All domains, {len(raw_samples)} samples")

        samples = []
        for sample in raw_samples:
            question = sample.get("question", "")
            steps = sample.get("steps", [])
            labels = sample.get("labels", [])
            for step_text, label in zip(steps, labels):
                text = f"{question}\n{step_text}"
                encoded = tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=512,
                    return_tensors=None,
                )
                samples.append(
                    {
                        "input_ids": torch.tensor(
                            encoded["input_ids"], dtype=torch.long
                        ),
                        "attention_mask": torch.tensor(
                            encoded["attention_mask"], dtype=torch.long
                        ),
                        "label": float(label),
                    }
                )
    else:
        loader = PRM800KLoader(data_dir=args.data_dir, split=args.split)
        try:
            samples = loader.build_step_dataset(tokenizer=tokenizer)
        except FileNotFoundError:
            print(f"[ERROR] Data not found at {args.data_dir}")
            return

    dataloader = DataLoader(
        samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_step_batch,
    )

    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            predictions = model(input_ids, attention_mask)
            loss = F.mse_loss(predictions, labels)
            total_loss += loss.item()
            num_batches += 1

            # Binary accuracy (threshold at 0.5)
            pred_binary = (predictions > 0.5).float()
            label_binary = (labels > 0.5).float()
            correct += (pred_binary == label_binary).sum().item()
            total += labels.numel()

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / total if total > 0 else 0.0

    print(f"[EVAL] Results:")
    print(f"  MSE Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
