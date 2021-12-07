# Adapted from: https://github.com/lucidrains/perceiver-pytorch/edit/main/perceiver_pytorch/perceiver_io.py

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
import perceiver_pytorch
from perceiver_pytorch.perceiver_io import *#PreNorm, Attention, FeedForward
from einops import rearrange, repeat

import sys
sys.path.append('/nfs/stak/users/popeq/Research/Microbiome/vilbert-multi-task/vilbert')
import vilbert

class CrossModalAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        num_latents
    ):
        super().__init__()
        self.value = nn.Parameter(torch.randn(num_latents, emb_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(
        self,
        key,
        query
    ):
        batch_size = key.shape[0]
        sa_value = self.value.unsqueeze(1).repeat(1, batch_size, 1)

        attn_output, attn_output_weights = multihead_attn(query, key, sa_value)
        return attn_output

class PerceiverIOTwoChannel(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()
        #self.vlibert_cfg = vilbert.BertConfig(logits_dim, hidden_size=latent_dim, bi_hidden_size=latent_dim, hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        #                                      v_attention_probs_dropout_prob=0, v_hidden_dropout_prob=0, bi_num_attention_heads=8, v_hidden_size=latent_dim)
        #self.cross_modal_attention = vilbert.BertBiAttention(self.vlibert_cfg)

        self.latents_1 = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attend_blocks_1 = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        self.decoder_cross_attn_1 = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff_1 = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.latents_2 = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attend_blocks_2 = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        self.decoder_cross_attn_2 = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff_2 = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args),
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args),
            ]))




        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def common_forward(
        self,
        latents,
        cross_attend_blocks,
        data,
        mask = None,
    ):
        b, *_, device = *data.shape, data.device
        x = repeat(latents, 'n d -> b n d', b = b)
        cross_attn, cross_ff = cross_attend_blocks

        # cross attention only happens once for Perceiver IO

        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x
        return x, b


    def forward(
        self,
        data_1,
        data_2,
        mask = None,
        ca_mask_1 = None,
        ca_mask_2 = None,
        queries = None
    ):
        x_1, b = self.common_forward(self.latents_1, self.cross_attend_blocks_1, data_1, ca_mask_1)
        x_2, _ = self.common_forward(self.latents_2, self.cross_attend_blocks_2, data_2, ca_mask_2)
        print(data_1.size(), data_2.size(), b)
        if not exists(ca_mask_1):
            ca_mask_1 = torch.ones_like(x_1[:,:,0])
        if not exists(ca_mask_2):
            ca_mask_2 = torch.ones_like(x_2[:,:,0])
        # layers

        for self_attn_1, self_ff_1, self_attn_2, self_ff_2 in self.layers:
            x_1 = self_attn_1(x_1) + x_1
            x_1 = self_ff_1(x_1) + x_1
            
            x_2 = self_attn_2(x_2) + x_2
            x_2 = self_ff_2(x_2) + x_2

            print(x_1.size(), x_2.size(), ca_mask_1.size(), ca_mask_2.size())

            x_2, x_1, _ = self.cross_modal_attention(x_1, ca_mask_1, x_2, ca_mask_2)
        x = torch.cat((x_1, x_2), dim=1)

        if not exists(queries):
            return x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents
        
        latents = self.decoder_cross_attn(queries, context = x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out

        return self.to_logits(latents)



class PerceiverLMTwoChannel(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver_io = PerceiverIOTwoChannel(
            dim = dim,
            queries_dim = dim,
            logits_dim = num_tokens,
            **kwargs
        )

    def forward(
        self,
        x_1,
        x_2,
        mask = None,
        ca_mask_1 = None,
        ca_mask_2 = None
        
    ):
        n_1, n_2, device = x_1.shape[1], x_2.shape[1], x_1.device
        x_1 = self.token_emb(x_1)
        x_2 = self.token_emb(x_2)

        pos_emb_1 = self.pos_emb(torch.arange(n_1, device = device))
        pos_emb_1 = rearrange(pos_emb_1, 'n d -> () n d')
        x_1 = x_1 + pos_emb_1

        pos_emb_2 = self.pos_emb(torch.arange(n_2, device = device))
        pos_emb_2 = rearrange(pos_emb_2, 'n d -> () n d')
        x_2 = x_2 + pos_emb_2


        queries = torch.cat((x_1, x_2), dim=1)
        logits = self.perceiver_io(x_1, x_2, mask = mask, ca_mask_1 = ca_mask_1, ca_mask_2 = ca_mask_2, queries = queries)
        return logits
