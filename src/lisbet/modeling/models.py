"""Multi-task model for different tasks."""

from typing import Any

import torch
from torch import nn

from lisbet.modeling.backbones.base import BackboneInterface


class MultiTaskModel(nn.Module):
    """Multi-task model that combines a backbone with multiple task-specific heads.

    This model enables training and inference across multiple tasks using a shared
    backbone representation. Each task has its own dedicated head that processes
    the backbone output.

    Parameters
    ----------
    backbone : BackboneInterface
        The backbone model that processes input sequences and produces
        shared representations.
    task_heads : dict[str, nn.Module]
        Dictionary mapping task IDs to their corresponding task-specific heads.

    Attributes
    ----------
    backbone : BackboneInterface
        The shared backbone model.
    task_heads : nn.ModuleDict
        Dictionary of task-specific heads.
    """

    def __init__(
        self,
        backbone: BackboneInterface,
        task_heads: dict[str, nn.Module],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        """Forward pass through the model for a specific task.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim).
        task_id : str
            Identifier for the task to use. Must be a key in task_heads.

        Returns
        -------
        torch.Tensor
            Task-specific output tensor. Shape depends on the specific task head.

        Raises
        ------
        KeyError
            If task_id is not found in the available task heads.
        """
        x = self.backbone(x)
        x = self.task_heads[task_id](x)
        return x

    def get_task_ids(self) -> list[str]:
        """Get the list of available task IDs.

        Returns
        -------
        list[str]
            List of task IDs that can be used with this model.
        """
        return list(self.task_heads.keys())

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary for this model.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary containing backbone config and task head configs.
        """
        task_head_configs = {}
        for task_id, head in self.task_heads.items():
            if hasattr(head, "get_config"):
                task_head_configs[task_id] = {
                    "type": head.__class__.__name__,
                    "config": head.get_config(),
                }
            else:
                # Fallback for heads without get_config method
                task_head_configs[task_id] = {
                    "type": head.__class__.__name__,
                    "config": {},
                }

        return {
            "backbone": {
                "type": self.backbone.__class__.__name__,
                "config": self.backbone.get_config(),
            },
            "task_heads": task_head_configs,
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        backbone_registry: dict[str, type] | None = None,
        head_registry: dict[str, type] | None = None,
    ) -> "MultiTaskModel":
        """Create a MultiTaskModel instance from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary containing backbone and task head configs.
        backbone_registry : dict[str, type] or None, optional
            Registry mapping backbone type names to their classes.
            If None, uses a default registry.
        head_registry : dict[str, type] or None, optional
            Registry mapping head type names to their classes.
            If None, uses a default registry.

        Returns
        -------
        MultiTaskModel
            New MultiTaskModel instance created from the configuration.

        Raises
        ------
        ValueError
            If backbone or head types are not found in the registries.
        """
        # Default registries
        if backbone_registry is None:
            from lisbet.modeling.backbones.transformer import TransformerBackbone

            backbone_registry = {
                "TransformerBackbone": TransformerBackbone,
            }

        if head_registry is None:
            from lisbet.modeling.heads.classification import (
                FrameClassificationHead,
                WindowClassificationHead,
            )
            from lisbet.modeling.heads.embedding import EmbeddingHead

            head_registry = {
                "FrameClassificationHead": FrameClassificationHead,
                "WindowClassificationHead": WindowClassificationHead,
                "EmbeddingHead": EmbeddingHead,
            }

        # Create backbone
        backbone_config = config["backbone"]
        backbone_type = backbone_config["type"]
        if backbone_type not in backbone_registry:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        backbone_cls = backbone_registry[backbone_type]
        backbone = backbone_cls.from_config(backbone_config["config"])

        # Create task heads
        task_heads = {}
        for task_id, head_config in config["task_heads"].items():
            head_type = head_config["type"]
            if head_type not in head_registry:
                raise ValueError(f"Unknown head type: {head_type}")

            head_cls = head_registry[head_type]
            task_heads[task_id] = head_cls.from_config(head_config["config"])

        return cls(backbone=backbone, task_heads=task_heads)
