"""Single-machine multi-process federated simulation scheduler."""

from typing import Optional

import torch
import torch.nn as nn

from fclprm.federated.client import FederatedClient
from fclprm.federated.server import FederatedServer
from fclprm.utils.seed import set_seed


class FederatedSimulator:
    """Simulate multiple federated clients on a single machine.

    Uses sequential client execution (not true multiprocessing) for simplicity.
    Each client is trained one after another on the same device.
    """

    def __init__(
        self,
        num_clients: int,
        num_rounds: int,
        global_model: nn.Module,
        client_data: list[list[dict]],
        aggregation_rule: str = "fedavg",
        seed: int = 42,
        anchor_inputs: dict | None = None,
        anchor_steps: list[str] | None = None,
        dp_enabled: bool = False,
        dp_epsilon: float = 4.0,
        dp_delta: float = 1e-5,
    ) -> None:
        """Initialize simulator.

        Args:
            num_clients: Number of simulated clients.
            num_rounds: Total federated training rounds.
            global_model: Initial global model.
            client_data: List of data splits, one per client.
            aggregation_rule: Aggregation strategy name.
            seed: Random seed.
            anchor_inputs: Optional dict with pre-tokenized anchor inputs:
                {"input_ids": LongTensor (N, L), "attention_mask": LongTensor
                (N, L)}. Required when aggregation_rule == "anchor_prm".
            anchor_steps: Anchor step texts (logged for reproducibility,
                not used in math).
            dp_enabled: Whether to enable DP-SGD on clients.
            dp_epsilon: Privacy budget epsilon (if DP enabled).
            dp_delta: Privacy budget delta (if DP enabled).
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.global_model = global_model
        self.client_data = client_data
        self.aggregation_rule = aggregation_rule
        self.seed = seed
        self.anchor_inputs = anchor_inputs
        self.anchor_steps = anchor_steps or []
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta

        if aggregation_rule == "anchor_prm" and anchor_inputs is None:
            raise ValueError(
                "anchor_prm aggregation requires anchor_inputs (pre-tokenized "
                "input_ids + attention_mask) to compute per-client embeddings"
            )

        self.server = FederatedServer(
            global_model=global_model,
            aggregation_rule=aggregation_rule,
            anchor_steps=self.anchor_steps,
        )

        self.clients: list[FederatedClient] = []
        for i in range(num_clients):
            client_model = (
                type(global_model)(
                    backbone=global_model.backbone,
                    head_dim=global_model.head.head_dim,
                )
                if hasattr(global_model, "backbone")
                else global_model
            )

            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_data=client_data[i],
                dp_enabled=dp_enabled,
                dp_epsilon=dp_epsilon,
                dp_delta=dp_delta,
            )
            self.clients.append(client)

        self.history: list[dict] = []

    def _extract_anchor_embeddings(
        self, client: FederatedClient, device: str
    ) -> torch.Tensor:
        """Run the (post-trained) client model over anchor inputs.

        Returns post-ReLU head features of shape (N, head_dim) on CPU,
        ready to ship to the server-side aligner.
        """
        client.model.to(device)
        client.model.eval()
        input_ids = self.anchor_inputs["input_ids"].to(device)
        attention_mask = self.anchor_inputs["attention_mask"].to(device)
        with torch.no_grad():
            embs = client.model.get_head_embedding(input_ids, attention_mask)
        return embs.detach().cpu()

    def run(
        self,
        local_epochs: int = 2,
        local_batch_size: int = 32,
        local_lr: float = 1e-4,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
    ) -> dict:
        """Run the full federated training simulation.

        Args:
            local_epochs: Number of local epochs per client per round.
            local_batch_size: Local training batch size.
            local_lr: Local learning rate.
            device: Device for training.
            max_grad_norm: Per-sample gradient clipping bound (used when DP enabled).

        Returns:
            Dict of training history and final metrics.
        """
        set_seed(self.seed)

        for round_num in range(self.num_rounds):
            # 1. Broadcast global model to all clients
            global_state = self.server.broadcast()
            for client in self.clients:
                client.model.load_state_dict(global_state)

            # 2. Local training on each client
            client_updates = []
            round_losses = []
            for client in self.clients:
                update = client.local_train(
                    num_epochs=local_epochs,
                    batch_size=local_batch_size,
                    learning_rate=local_lr,
                    device=device,
                    max_grad_norm=max_grad_norm,
                )
                if self.aggregation_rule == "anchor_prm":
                    update["anchor_embeddings"] = self._extract_anchor_embeddings(
                        client, device=device
                    )
                client_updates.append(update)
                round_losses.append(update["loss"])

            # 3. Server aggregation
            self.server.aggregate(client_updates)

            avg_loss = sum(round_losses) / len(round_losses)
            self.history.append(
                {
                    "round": round_num,
                    "avg_loss": avg_loss,
                    "client_losses": round_losses,
                }
            )

        return {
            "history": self.history,
            "final_model": self.server.get_global_model(),
            "num_rounds": self.num_rounds,
            "num_clients": self.num_clients,
        }
