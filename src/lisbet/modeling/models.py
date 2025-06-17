"""Multi-task model for different tasks."""

from torch import nn


class MultiTaskModel(nn.Module):
    """Base model for all tasks."""

    def __init__(self, backbone, task_heads):
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(self, x, task_id):
        x = self.backbone(x)
        x = self.task_heads[task_id](x)
        return x
