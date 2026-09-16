import torch
from torch import nn
from torch.nn import functional as F


class StackAttention(nn.Module):
    def __init__(self, d=768, k=512, dropout=True):
        super().__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        self.dropout = nn.Dropout(p=0.5) if dropout else nn.Identity()
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        hidden = torch.tanh(self.ff_image(vi) + self.ff_ques(vq))
        scores = self.ff_attention(self.dropout(hidden)).squeeze(2)
        weights = F.softmax(scores, dim=1)
        return (weights.unsqueeze(2) * vi).sum(1) + vq.squeeze(1)
