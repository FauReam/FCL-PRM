"""Single federated client: local PRM training + optional DP-SGD."""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FederatedClient:
    """Client-local training for PRM.

    Responsibilities:
        - Load local data partition
        - Train PRM head (backbone frozen)
        - Apply DP-SGD if privacy is enabled
        - Return model delta + step embeddings for aggregation
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: list[dict],
        dp_enabled: bool = False,
        dp_epsilon: float = 4.0,
        dp_delta: float = 1e-5,
    ) -> None:
        """Initialize client.

        Args:
            client_id: Unique client identifier.
            model: Local model instance.
            train_data: Client's local training data (list of step samples).
            dp_enabled: Whether to apply DP-SGD during local training.
            dp_epsilon: Privacy budget epsilon (if DP enabled).
            dp_delta: Privacy budget delta (if DP enabled).
        """
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta

    def local_train(
        self,
        num_epochs: int,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
    ) -> dict:
        """Run local training for specified epochs.

        Args:
            num_epochs: Number of local epochs per round.
            batch_size: Local batch size.
            learning_rate: Local learning rate.
            device: Device for training.
            max_grad_norm: Per-sample gradient clipping bound (used when DP enabled).

        Returns:
            Dict containing model state dict and training metrics.
        """
        self.model.to(device)
        self.model.train()

        # Only optimize parameters that require grad (PRM head)
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        # Simple dataloader from list of dicts
        from fclprm.data.utils import collate_step_batch

        loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_step_batch,
        )

        # Apply DP-SGD if enabled
        model = self.model
        if self.dp_enabled:
            from fclprm.federated.dp import StepLevelDPSGD

            dp_engine = StepLevelDPSGD(
                epsilon=self.dp_epsilon,
                delta=self.dp_delta,
                max_grad_norm=max_grad_norm,
            )
            model, optimizer, loader = dp_engine.make_private(
                model=model,
                optimizer=optimizer,
                data_loader=loader,
                epochs=num_epochs,
            )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                predictions = model(input_ids, attention_mask)
                loss = F.mse_loss(predictions, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Opacus wraps the model in GradSampleModule, which prefixes
        # state_dict keys with "_module.". Strip the prefix before
        # shipping to the server so aggregators can match parameter names.
        raw_model = model._module if hasattr(model, "_module") else model
        return {
            "client_id": self.client_id,
            "state_dict": {k: v.cpu().clone() for k, v in raw_model.state_dict().items()},
            "loss": avg_loss,
            "num_samples": len(self.train_data),
        }

    def get_step_embeddings(
        self,
        steps: list[str],
        tokenizer,
        device: str = "cuda",
        max_length: int = 512,
    ) -> torch.Tensor:
        """Extract step embeddings from local model for CD-SPI.

        Args:
            steps: List of step strings.
            tokenizer: HuggingFace tokenizer.
            device: Device for inference.
            max_length: Max token length.

        Returns:
            Step embeddings tensor of shape (len(steps), hidden_dim).
        """
        self.model.to(device)
        self.model.eval()

        encoded = tokenizer(
            steps,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            embeddings = self.model.get_step_embedding(input_ids, attention_mask)

        return embeddings.cpu()
