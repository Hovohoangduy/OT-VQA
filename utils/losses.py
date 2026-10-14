"""Answer-generation losses with explicit token or question normalization."""

from torch import nn
from torch.nn import functional as F


class AnswerCrossEntropyLoss(nn.Module):
    """Optionally give short and long answers equal weight within a batch.

    Inputs are vocabulary logits [B,V,T] and target IDs [B,T]. Evaluation
    continues to use conventional token-averaged CE for comparable reports.
    """

    def __init__(self, ignore_index, label_smoothing=0.0, normalization="token"):
        super().__init__()
        if normalization not in {"token", "example"}:
            raise ValueError("Loss normalization must be token or example")
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.normalization = normalization

    def forward(self, logits, targets):
        if self.normalization == "token":
            return F.cross_entropy(logits, targets, ignore_index=self.ignore_index,
                                   label_smoothing=self.label_smoothing)
        counts = targets.ne(self.ignore_index).sum(dim=1)
        if counts.eq(0).any():
            raise ValueError("Each answer needs at least one non-padding target")
        losses = F.cross_entropy(logits, targets, ignore_index=self.ignore_index,
                                 label_smoothing=self.label_smoothing, reduction="none")
        return (losses.sum(dim=1) / counts).mean()
