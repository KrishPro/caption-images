"""
Written by KrishPro @ KP

filename: `model.py`
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor = None):
    """
    Q.shape: (B, QS, E)
    K.shape: (B, KS, E)
    V.shape: (B, KS, E)
    attn_mask: (B, QS, KS)

    returns: (B, QS, E)
    """

    B, QS, E = Q.shape
    B, KS, E = K.shape

    if attn_mask is None: attn_mask = torch.zeros(B, QS, KS, device=Q.device)

    energy = torch.nan_to_num(F.softmax(torch.baddbmm(attn_mask, Q, K.transpose(-2, -1)) / (E ** 0.5), dim=2))
    # energy.shape: (B, QS, KS)

    return torch.bmm(energy, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, bias:bool = False) -> None:
        super().__init__()
        
        assert (d_model % n_heads) == 0, "d_model should be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads

        self.Q = nn.Linear(d_model, d_model, bias=bias)
        self.K = nn.Linear(d_model, d_model, bias=bias)
        self.V = nn.Linear(d_model, d_model, bias=bias)

        self.WO = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, xa: torch.Tensor = None, attn_mask: torch.Tensor = None):
        """
        x.shape: (B, S, E)
        xa.shape: (B, SS, E)
        attn_mask: (B, S, SS)

        returns: (B, S, E)
        """

        B, S, E = x.shape

        assert (xa is None or type(xa) == dict) or B == xa.size(0), "Batchsize should be same for x and xa"
        assert ((xa is None or type(xa) == dict) or E == xa.size(2)) and E == self.d_model, "Embed_dims should be same for x and xa and be equal to d_model"

        Q: torch.Tensor = self.Q(x)

        if type(xa) == dict:
            K: torch.Tensor = xa[self.K]
            V: torch.Tensor = xa[self.V]

        else:
            K: torch.Tensor = self.K(xa if xa is not None else x)
            V: torch.Tensor = self.V(xa if xa is not None else x)
        
        SS = K.size(1)

        Q = Q.view(B,  S, self.n_heads, self.d_heads).transpose(1, 2).reshape(B*self.n_heads,  S, self.d_heads)
        K = K.view(B, SS, self.n_heads, self.d_heads).transpose(1, 2).reshape(B*self.n_heads, SS, self.d_heads)
        V = V.view(B, SS, self.n_heads, self.d_heads).transpose(1, 2).reshape(B*self.n_heads, SS, self.d_heads)

        attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0) if (attn_mask is not None) and attn_mask.dim() == 3 else attn_mask
        # attn_mask = (B*H, S, SS)

        out = self_attention(Q, K, V, attn_mask=attn_mask).reshape(B, self.n_heads, S, self.d_heads).transpose(1, 2).reshape(B, S, self.n_heads*self.d_heads)

        return self.WO(out)

class Encoder(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, dropout_p:float) -> None:
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:

        x = self.norm1(x + self.dropout(self.attn(x, attn_mask=attn_mask)))

        x = self.norm2(x + self.dropout(self.feedforward(x)))

        return x

class Decoder(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, dropout_p:float) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, xa: torch.Tensor, attn_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        
        x = self.norm1(x + self.dropout(self.attn(x, attn_mask=attn_mask)))
        
        x = self.norm2(x + self.dropout(self.cross_attn(x, xa, attn_mask=memory_mask)))
        
        x = self.norm3(x + self.dropout(self.feedforward(x)))

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model:int, dropout_p:float=0.1, max_len:float=5000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pos_embeddings', self.generate_sinusoids(max_len, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, embeddings: torch.Tensor):
        return self.dropout((embeddings * (self.d_model ** 0.5)) + self.pos_embeddings[:embeddings.size(1)])

    def generate_sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)



class ViT(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, num_layers:int, tgt_vocab_size:int, dropout_p:float=0.1, pad_idx:int=0) -> None:
        super().__init__()

        self.max_len = 5_000
        self.pad_idx = pad_idx
        self.mask_token = -1e+25 # Usually `float('-inf')`, But i used -1e+25 for numerical stablity
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_embedding = PositionalEmbedding(d_model, dropout_p, max_len=self.max_len)
        self.register_buffer('lm_mask', torch.empty(self.max_len, self.max_len).fill_(self.mask_token).triu_(1), persistent=False)

        self.input = nn.Sequential(
            nn.Conv2d(3, 3*(dim_feedforward//d_model), kernel_size=3, padding=1),
            nn.Conv2d(3*(dim_feedforward//d_model), 1, kernel_size=3, padding=1)
        )

        self.encoder_layers = nn.ModuleList([Encoder(d_model, n_heads, dim_feedforward, dropout_p) for _ in range(num_layers)])
        
        self.decoder_layers = nn.ModuleList([Decoder(d_model, n_heads, dim_feedforward, dropout_p) for _ in range(num_layers)])

        self.output = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.output.weight = self.tgt_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @classmethod
    def from_ckpt(cls, ckpt_path: str):
        ckpt = torch.load(ckpt_path)
        
        model = cls(**ckpt['dims'])

        model.load_state_dict(ckpt['state_dict'])

        return model

    @classmethod
    def add_method(cls, function):
        setattr(cls, function.__name__, function)
        return function

    def forward(self, src: torch.Tensor, tgt:torch.Tensor):
        """
        src.shape: (B, S)
        tgt.shape: (B, T)

        returns: (B, T, TV)
        """

        # Generating masks
        tgt_mask = torch.zeros(*tgt.shape, device=tgt.device).masked_fill(tgt == self.pad_idx, self.mask_token)
        
        memory_mask = tgt_mask.unsqueeze(2)
        tgt_mask = tgt_mask.unsqueeze(1) + tgt_mask.unsqueeze(2)


        tgt_mask += self.lm_mask[:tgt.size(1), :tgt.size(1)]

        src: torch.Tensor = self.input(src)

        src: torch.Tensor = src.reshape([src.size(0), 16, 16, 16, 16]).permute(0, 1, 3, 2, 4).reshape([src.size(0), 256, 256])

        # src.shape = (B, 1, 256, 256)
        
        # Embedding src & tgt
        src: torch.Tensor = self.pos_embedding(src)
        tgt: torch.Tensor = self.pos_embedding(self.tgt_embedding(tgt))

        # Encoding src
        for layer in self.encoder_layers:
            src = layer(src)

        # Decoding tgt
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, attn_mask=tgt_mask, memory_mask=memory_mask)


        # Applying a linear layer and returning
        return self.output(tgt)


def test(use_cuda=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    print(f"DEVICE: {device}")

    transformer = ViT(256, 4, 1024, 3, 30_000, 30_000).to(device)

    src = torch.randn(8, 3, 256, 256, device=device)
    tgt = torch.randint(low=0, high=5, size=(8, 75), device=device)

    print(transformer(src, tgt).shape)

if __name__ == '__main__':
    test()