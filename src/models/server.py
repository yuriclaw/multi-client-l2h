"""Server model with per-client LoRA adapters.

The server backbone (ViT or ResNet via timm) is frozen. Each client gets
a lightweight LoRA adapter injected via HuggingFace PEFT, enabling
personalized server inference without full fine-tuning.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model, PeftModel


class ServerWithLoRA(nn.Module):
    """Frozen backbone + switchable per-client LoRA adapters."""

    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        num_classes: int = 10,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["qkv", "proj"],
            bias="none",
        )

        # Create and freeze the backbone
        base_model = timm.create_model(
            backbone_name, pretrained=True, num_classes=num_classes
        )
        for p in base_model.parameters():
            p.requires_grad = False

        # Wrap with PEFT — creates a default adapter
        self.model = get_peft_model(base_model, self.lora_config, adapter_name="default")
        self._adapter_names: set[str] = {"default"}
        self._active_adapter: str = "default"

    # ------------------------------------------------------------------
    # Per-client adapter management
    # ------------------------------------------------------------------

    def add_adapter(self, client_id: str) -> None:
        """Create a new LoRA adapter for a client (if not exists)."""
        if client_id in self._adapter_names:
            return
        self.model.add_adapter(client_id, self.lora_config)
        self._adapter_names.add(client_id)

    def set_adapter(self, client_id: str) -> None:
        """Switch the active LoRA adapter."""
        if client_id not in self._adapter_names:
            raise KeyError(f"Adapter '{client_id}' not found. Call add_adapter first.")
        self.model.set_adapter(client_id)
        self._active_adapter = client_id

    def save_adapter(self, client_id: str, save_dir: str) -> None:
        """Save a single client's adapter weights."""
        self.set_adapter(client_id)
        path = os.path.join(save_dir, client_id)
        self.model.save_pretrained(path)

    def load_adapter(self, client_id: str, load_dir: str) -> None:
        """Load adapter weights from disk."""
        path = os.path.join(load_dir, client_id)
        self.model.load_adapter(path, adapter_name=client_id)
        self._adapter_names.add(client_id)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, client_id: Optional[str] = None
    ) -> torch.Tensor:
        """Forward pass using the specified client's adapter.

        Args:
            x: Input tensor (B, C, H, W).
            client_id: Which adapter to use. If None, uses current active.

        Returns:
            Logits of shape (B, num_classes).
        """
        if client_id is not None and client_id != self._active_adapter:
            self.set_adapter(client_id)
        return self.model(x)

    def adapter_parameters(self, client_id: Optional[str] = None) -> list[nn.Parameter]:
        """Return trainable parameters for the given adapter."""
        if client_id is not None:
            self.set_adapter(client_id)
        return [p for p in self.model.parameters() if p.requires_grad]
