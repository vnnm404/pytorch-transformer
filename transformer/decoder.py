import torch
from torch import nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, use_kv_cache=False):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, nhead, use_kv_cache=use_kv_cache)
        self.self_attention_layer_norm = nn.LayerNorm(d_model)

        self.encoder_decoder_attention = MultiHeadAttention(d_model, nhead, use_kv_cache=use_kv_cache)
        self.encoder_decoder_layer_norm = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_ffn)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, tgt, tgt_mask=None, mem_mask=None, kv_cache=None):
        # src: [batch_size, src_sequence_length, d_model]
        # tgt: [batch_size, tgt_sequence_length, d_model]
        # tgt_mask: [tgt_sequence_length, tgt_sequence_length]
        # mem_mask: [tgt_sequence_length, src_sequence_length]
        # kv_cache: dict[str, Tensor]  always has "key" and "value" entries if not None

        if tgt is None:
            tgt = src
        else:
            residual = tgt
            tgt, _ = self.self_attention(query=tgt, key=tgt, value=tgt, mask=tgt_mask)
            tgt = self.self_attention_layer_norm(tgt + residual)

        residual = tgt
        tgt, _ = self.encoder_decoder_attention(
            query=tgt, key=src, value=src, mask=mem_mask, kv_cache=kv_cache
        )
        tgt = self.encoder_decoder_layer_norm(tgt + residual)

        residual = tgt
        tgt = self.feed_forward(tgt)
        tgt = self.feed_forward_layer_norm(tgt + residual)

        return tgt


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, d_ffn):
        super(Decoder, self).__init__()

        self.decoders = nn.ModuleList(
            [DecoderLayer(d_model, nhead, d_ffn) for _ in range(num_decoder_layers)]
        )

    def forward(self, src, tgt=None, tgt_mask=None, mem_mask=None, kv_cache=None):
        # src: [batch_size, src_sequence_length, d_model]
        # tgt: [batch_size, tgt_sequence_length, d_model]
        # tgt_mask: [tgt_sequence_length, tgt_sequence_length]
        # mem_mask: [tgt_sequence_length, src_sequence_length]

        for decoder in self.decoders:
            tgt = decoder(src, tgt, tgt_mask, mem_mask, kv_cache)

        return tgt
