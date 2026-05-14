#!/usr/bin/env python3
"""M3-M6: Federated PRM simulation main entry point.

Usage:
    python scripts/run_federated.py --config configs/m3_naive_fedavg.yaml
    python scripts/run_federated.py --config configs/m4_anchor_prm.yaml
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from fclprm.data.versa_loader import VersaPRMLoader
from fclprm.federated.simulator import FederatedSimulator
from fclprm.models.base_wrapper import StepRewardModel
from fclprm.utils.config import ExperimentConfig
from fclprm.utils.logging import ExperimentLogger
from fclprm.utils.seed import set_seed


def _load_hf_asset(load_fn, model_name: str, **kwargs):
    """Load from HF Hub with automatic fallback to local cache on network errors."""
    try:
        return load_fn(model_name, local_files_only=False, **kwargs)
    except Exception as e:
        err_name = type(e).__name__
        if any(x in err_name for x in ("ConnectTimeout", "ConnectionError", "HTTPError", "OfflineModeIsEnabled")):
            print(f"[WARN] Hub unreachable ({err_name}), falling back to local cache...")
            return load_fn(model_name, local_files_only=True, **kwargs)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run federated PRM simulation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    config = ExperimentConfig(args.config)
    set_seed(config.get("experiment.seed", 42))

    device = config.get(
        "hardware.device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    log_dir = config.get("logging.log_dir", "./logs")

    logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_id=config.get("experiment.name", "federated_prm"),
    )

    print(
        f"[{config.get('experiment.milestone')}] Initializing model: {config.get('model.backbone')}"
    )
    try:
        tokenizer = _load_hf_asset(
            AutoTokenizer.from_pretrained, config.get("model.backbone")
        )
    except OSError as e:
        print(f"[ERROR] Failed to load tokenizer for '{config.get('model.backbone')}'.")
        print(f"  {e}")
        print("  Please ensure the model name is correct and you have internet access,")
        print(
            "  or download the model locally and set local_files_only=True in config."
        )
        return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        backbone = _load_hf_asset(
            AutoModel.from_pretrained,
            config.get("model.backbone"),
            dtype=torch.float32,
        )
    except OSError as e:
        print(f"[ERROR] Failed to load backbone for '{config.get('model.backbone')}'.")
        print(f"  {e}")
        print("  Please ensure the model name is correct and you have internet access,")
        print(
            "  or download the model locally and set local_files_only=True in config."
        )
        return
    global_model = StepRewardModel(
        backbone=backbone,
        head_dim=config.get("model.prm_head_dim", 256),
    )

    print(
        f"[{config.get('experiment.milestone')}] Loading data from: {config.get('data.data_dir')}"
    )
    versa_loader = VersaPRMLoader(
        data_dir=config.get("data.data_dir"),
    )

    try:
        versa_loader.load()
    except FileNotFoundError as e:
        print(f"[ERROR] Data not found: {e}")
        print("Please download VersaPRM data to the specified data_dir.")
        return

    # Build step-level datasets for each client domain
    num_clients = config.get("federated.num_clients", 4)
    domains = config.get("data.domains", ["math", "code", "medical", "general"])
    max_length = config.get("data.max_length", 512)
    samples_per_client = config.get("data.samples_per_client", 5000)

    client_data = []
    for i in range(num_clients):
        domain = domains[i % len(domains)]
        domain_samples = versa_loader.load_domain(domain)

        # Tokenize steps
        step_samples = []
        for sample in domain_samples[:samples_per_client]:
            question = sample.get("question", "")
            steps = sample.get("steps", [])
            labels = sample.get("labels", [])

            if len(steps) != len(labels):
                raise ValueError(
                    f"steps/labels length mismatch in domain '{domain}': "
                    f"{len(steps)} steps vs {len(labels)} labels"
                )
            for step_text, label in zip(steps, labels):
                text = f"{question}\n{step_text}"
                encoded = tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None,
                )
                step_samples.append(
                    {
                        "input_ids": torch.tensor(encoded["input_ids"]),
                        "attention_mask": torch.tensor(encoded["attention_mask"]),
                        "label": float(label),
                    }
                )

        client_data.append(step_samples)
        print(f"  Client {i} ({domain}): {len(step_samples)} step samples")

    print(f"[{config.get('experiment.milestone')}] Starting federated simulation")
    print(f"  Rounds: {config.get('federated.num_rounds')}")
    print(f"  Clients: {num_clients}")
    print(f"  Aggregation: {config.get('federated.aggregation')}")

    # Anchor-PRM aggregation needs a small set of shared anchor steps to
    # extract per-client head embeddings. Use a generic, domain-agnostic
    # default so the experiment is reproducible without external assets;
    # production runs should override this via `anchor.steps` in the YAML.
    aggregation_rule = config.get("federated.aggregation", "fedavg")
    anchor_inputs = None
    anchor_steps = None
    if aggregation_rule == "anchor_prm":
        anchor_steps = config.get("anchor.steps", None) or [
            "Let x be the variable we want to solve for.",
            "First, identify the key constraints of the problem.",
            "Therefore, the answer must be positive.",
            "We can verify this by substitution.",
            "Combining the two equations gives us a single unknown.",
            "By the definition above, this implies the next inequality.",
            "Hence, the conclusion follows from the previous step.",
            "Note that this assumption holds only when the input is non-empty.",
        ]
        anchor_encoded = tokenizer(
            anchor_steps,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        anchor_inputs = {
            "input_ids": anchor_encoded["input_ids"],
            "attention_mask": anchor_encoded["attention_mask"],
        }
        print(f"  Anchor steps: {len(anchor_steps)}")

    # DP configuration
    dp_enabled = config.get("dp.enabled", False)
    dp_epsilon = config.get("dp.epsilon", 4.0)
    dp_delta = config.get("dp.delta", 1e-5)
    dp_max_grad_norm = config.get("dp.max_grad_norm", 1.0)
    if dp_enabled:
        print(
            f"  DP-SGD: enabled (epsilon={dp_epsilon}, delta={dp_delta}, max_grad_norm={dp_max_grad_norm})"
        )

    simulator = FederatedSimulator(
        num_clients=num_clients,
        num_rounds=config.get("federated.num_rounds", 50),
        global_model=global_model,
        client_data=client_data,
        aggregation_rule=aggregation_rule,
        seed=config.get("experiment.seed", 42),
        anchor_inputs=anchor_inputs,
        anchor_steps=anchor_steps,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_delta=dp_delta,
    )

    results = simulator.run(
        local_epochs=config.get("federated.local_epochs", 2),
        local_batch_size=config.get("federated.local_batch_size", 32),
        local_lr=config.get("federated.local_learning_rate", 1e-4),
        device=device,
        max_grad_norm=dp_max_grad_norm,
    )

    print(f"[{config.get('experiment.milestone')}] Simulation complete")
    print(f"  Final avg loss: {results['history'][-1]['avg_loss']:.4f}")

    logger.log(
        milestone=config.get("experiment.milestone"),
        config_hash=config.hash(),
        metrics={
            "final_loss": results["history"][-1]["avg_loss"],
            "num_rounds": results["num_rounds"],
        },
    )


if __name__ == "__main__":
    main()
