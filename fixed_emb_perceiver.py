# Adapted from: https://github.com/lucidrains/perceiver-pytorch/edit/main/perceiver_pytorch/perceiver_io.py

from math import pi, log
from functools import wraps
from helpers import *

import torch
from torch import nn, einsum
import torch.nn.functional as F
import perceiver_pytorch
from perceiver_pytorch.perceiver_io import *#PreNorm, Attention, FeedForward
from performer_pytorch.performer_pytorch import FixedPositionalEmbedding
from einops import rearrange, repeat

# Perceiver LM with fixed positional embeddings:

class FixedPosPerceiverLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        pos_emb='fixed',
        dropout=0.0,
        num_classes=0,
        **kwargs
    ):
        super().__init__()
        self.pos_emb_type = pos_emb
        self.token_emb = nn.Embedding(num_tokens, dim)
        if pos_emb == 'fixed':
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len) 
        elif pos_emb == 'abs':
            self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver_io = PerceiverIO(
            dim = dim,
            queries_dim = dim,
            logits_dim = num_tokens,
            **kwargs
        )
        self.num_classes = num_classes
        if num_classes > 0:
            self.logits_to_classes = torch.nn.Linear(num_tokens, num_classes)

    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        if self.pos_emb_type == 'abs':
            pos_emb = self.pos_emb(torch.arange(n, device = device))
            pos_emb = rearrange(pos_emb, 'n d -> () n d')
        else:
            pos_emb = self.pos_emb(x)
        x = x + pos_emb

        logits = self.perceiver_io(x, mask = mask, queries = x)
        if self.num_classes > 0:
            class_logits = self.logits_to_classes(logits[:, 0, :])
            return logits, class_logits
        return logits
