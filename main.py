import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_encoders=6, num_decoders=6):
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, num_encoders)
        self.decoder = Decoder(d_model, num_heads, num_decoders)

    def forward(self, source, target, source_mask, target_mask):
        encoding = self.encoder(source, source_mask)
        return self.decoder(target, encoding, source_mask, target_mask)


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_encoders):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads) for _ in range(num_encoders)]
        )

    def forward(self, source, source_mask):
        encoding = source
        for layer in self.layers:
            encoding = layer(encoding, source_mask)
        return encoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.3):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model, num_heads, dropout=dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attention_normalization = nn.LayerNorm(d_model)
        self.feedforward_normalization = nn.LayerNorm(d_model)

    def forward(self, source, source_mask):
        x = source
        x += self.attention(q=x, k=x, v=x, mask=source_mask)  # += implements skip connections
        x - self.attention_normalization(x)
        x += self.feedforward(x)
        return self.feedforward_normalization(x)


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_decoders):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(num_decoders)]
        )

    def forward(self, target, encoding, target_mask, encoding_mask):
        output = target
        for layer in self.layers:
            output = layer(output, encoding, target_mask, encoding_mask)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.3):
        super().__init__()

        self.masked_attention = MultiHeadedAttention(
            d_model, num_heads, dropout=dropout
        )
        self.attention = MultiHeadedAttention(
            d_model, num_heads, dropout=dropout
        )
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.masked_attention_normalization = nn.LayerNorm(d_model)
        self.attention_normalization = nn.LayerNorm(d_model)
        self.feedforward_normalization = nn.LayerNorm(d_model)

    def forward(self, target, encoding, target_mask, encoding_mask):
        x = target
        x += self.masked_attention(q=x, k=x, v=x, mask=target_mask)
        x = self.masked_attention_normalization(x)
        x += self.attention(q=x, k=encoding, v=encoding, mask=encoding_mask)
        x = self.attention_normalization(x)
        x += self.feedforward(x)
        return self.feedforward_normalization(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_output_size = self.d_model // self.num_heads
        self.attentions = nn.ModuleList(
            [
                SelfAttention(d_model, self.attention_output_size)
                for _ in range(self.num_heads)
            ]
        )

    def forward(self, q, k, v, mask):
        x = torch.cat(
            [layer(q, k, v, mask) for layer in self.attentions], dim=-1
        )
        return self.output(x)


class SelfAttention(nn.Module):
    def __init__(self, d_model, output_size, dropout=0.3):
        super().__init__()
        self.query = nn.Linear(d_model, output_size)
        self.key = nn.Linear(d_model, output_size)
        self.value = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]  # Biases??
        target_len = q.shape[1]
        sequence_len = k.shape[1]
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        dim_k = key.size(-1)  # torch layer size (last dimension)
        scores = torch.bmm(query, key.transpose(1, 2)) // np.sqrt(dim_k)

        if mask is not None:
            expanded_mask = mask[:, None, :].expand(bs, target_len, sequence_len)
            if mask == 'subsequent':  # For use in decoder during training
                subsequent_mask = 1 - torch.triu(  # Upper triangle of matrix
                    torch.ones((target_len, target_len), device=mask.device, dtype=torch.uint8), diagonal=1
                )
        scores = scores.masked_fill(expanded_mask == 0, -float("Inf"))
        if mask == 'subsequent':
            scores = scores.masked_fill(subsequent_mask == 0, -float("Inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, value)


# From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.) / d_model))  # Need to look into this
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])

