import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.layer import *

def subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    if len(seq.size()) == 2:
        bs, len_s = seq.size()
    else:
        bs, len_s, dim = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, src_vocab, n_layers, n_head, d_k, d_v,
            d_model, d_hid, dropout=0.3, n_position=32, min_dist=32, max_dist=32, relative_pe=True):
        super(Encoder, self).__init__()

        #self.src_emb = Embeddings(src_vocab+1, d_model)
        #self.src_emb = nn.Linear(src_vocab, d_model)
        self.position_enc = PositionalEncoding(d_model, dropout, n_position=n_position)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_hid, n_head, d_k, d_v, min_dist, max_dist, dropout=dropout, relative_pe=relative_pe)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask=None):
        
        src = self.position_enc(src_seq)
        output = self.layer_norm(src)

        for layer in self.layer_stack:
            output, attn, attn_weights = layer(output, attn_mask=src_mask)

        return output, attn_weights


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, tgt_vocab, n_layers, n_head, d_k, d_v,
            d_model, d_hid, n_position=32, dropout=0.3, min_dist=32, max_dist=32, relative_pe=True):
        super(Decoder, self).__init__()

        self.tgt_emb = Embeddings(tgt_vocab+1, d_model)
        self.position_enc = PositionalEncoding(d_model, dropout, n_position=n_position)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_hid, n_head, d_k, d_v, min_dist, max_dist, dropout=dropout, relative_pe=relative_pe)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt_seq, enc_output, tgt_mask, src_mask):

        a_outs, a_weights = list(), list()

        tgt = self.position_enc(self.tgt_emb(tgt_seq))
        output = self.layer_norm(tgt)

        for dec_layer in self.layer_stack:
            output, attn, enc_attn, enc_attn_weights = dec_layer(
                output, enc_output, attn_mask=tgt_mask, enc_attn_mask=src_mask)
            a_outs.append(enc_attn)
            a_weights.append(enc_attn_weights)

        return output, a_outs, a_weights
