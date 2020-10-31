import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformer.attention import *

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, min_dist, max_dist, dropout=0.3, relative_pe=True):
        super(EncoderLayer, self).__init__()
        if relative_pe:
            self.attn = MultiHeadAttentionWithRPE(n_head, d_model, d_k, d_v, min_dist, max_dist, dropout=dropout)
        else:
            self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, src, attn_mask=None):
        output, attn, attn_weights = self.attn(
            src, src, src, mask=attn_mask)
        output = self.pos_ffn(output)
        return output, attn, attn_weights


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, min_dist, max_dist, dropout=0.3, relative_pe=True):
        super(DecoderLayer, self).__init__()
        if relative_pe:
            self.attn = MultiHeadAttentionWithRPE(n_head, d_model, d_k, d_v, min_dist, max_dist, dropout=dropout)
            self.enc_attn = MultiHeadAttentionWithRPE(n_head, d_model, d_k, d_v, min_dist, max_dist, dropout=dropout)
        else:
            self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, tgt, src,
            attn_mask=None, enc_attn_mask=None):
        output, attn, attn_weights = self.attn(
            tgt, tgt, tgt, mask=attn_mask)
        dec_output, enc_attn, enc_attn_weights = self.enc_attn(
            output, src, src, mask=enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, attn, enc_attn, enc_attn_weights


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(n_position, d_model)
        position = torch.arange(0, n_position).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model)).float()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)