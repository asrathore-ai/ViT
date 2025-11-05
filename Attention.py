import torch
import torch.nn as nn
import math 

from torch.nn.functional import softmax

# [B, H, seq_len, d_k], [B, H, seq_len, d_k], [B, H, seq_len, d_v]
def Attention(Q, K, V, masked=False):
    d_k = Q.shape[-1]
    scale = math.sqrt(d_k)
    scores = (Q @ K.transpose(-2, -1))/scale
    if masked:
        S = Q.shape[-2]
        mask = torch.triu(torch.ones([S,S] ,device=Q.device).bool(), diagonal=1)
        mask = mask.view(1, 1, S, S) # Broadcast across B,H
        scores.masked_fill_(mask, float("-inf"))
    o1 = softmax(scores, dim=-1)
    return o1 @ V
        

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, masked=False):
        super().__init__()
        self.masked = masked
        self.num_heads = num_heads
        d_k = d_model//num_heads 
        d_v = d_model//num_heads 
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.WO = nn.Linear(num_heads * d_v, d_model)

    
    def forward(self, Q, K, V):
        
        B, seq_len, d_model = Q.shape
        H = self.num_heads
        d_k = d_model // H 
        d_v = d_model // H

        Q_proj = self.Wq(Q)
        K_proj = self.Wk(K)
        V_proj = self.Wv(V)

        Q_batch = Q_proj.view(B, seq_len, H, d_k).permute( (0, 2, 1, 3) ) # (B, seq_len, H, d_k) -> (B, H, seq_len, d_k)
        K_batch = K_proj.view(B, seq_len, H, d_k).permute( (0, 2, 1, 3) ) # (B, seq_len, H, d_k) -> (B, H, seq_len, d_k)
        V_batch = V_proj.view(B, seq_len, H, d_v).permute( (0, 2, 1, 3) ) # (B, seq_len, H, d_v) -> (B, H, seq_len, d_v)

        # B, H, seq_len, d_v
        outs = Attention(Q_batch, K_batch, V_batch, self.masked)
        
        outs = outs.permute( (0, 2, 1, 3) )
        outs = outs.contiguous().view( B, seq_len, d_v * H)

        out = self.WO(outs)
        return out

class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model=512, inner_dim=2048):
        super().__init__()
        self.l1 = nn.Linear(d_model, inner_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(inner_dim, d_model)

    # [BatchSize, seq_len, d_model]
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.l2(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttentionLayer(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    # [BatchSize, seq_len, d_model]
    def forward(self, x):
        x = self.norm1(x + self.mha(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x

class Encoder(nn.Module):
    
    def __init__(self, d_model, num_heads, num_layers = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])

    # [BatchSize, seq_len, d_model]
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.maskedMHA = MultiHeadAttentionLayer(d_model, num_heads, masked=True)
        self.ln1 = nn.LayerNorm(d_model)

        self.mha = MultiHeadAttentionLayer(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.ffn = FeedForwardNetwork(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        x = self.ln1( x + self.maskedMHA(x, x, x))
        x = self.ln2( x + self.mha(x, encoder_output, encoder_output))
        x = self.ln3( x + self.ffn(x))
        return x
    
class Decoder(nn.Module):

    def __init__(self, d_model, num_heads, num_layers = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads) for _ in range(num_layers)
        ]) 

    # [BatchSize, seq_len, d_model]
    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x 

class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=512, num_heads=8):
        super().__init__()
        self.encoder = Encoder( d_model, num_heads )
        self.decoder = Decoder( d_model, num_heads )
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, d_model//2)
        deno = torch.pow(10_000, 2*i/d_model )
        args = pos/deno
        pe = torch.zeros([seq_len, d_model], dtype=torch.float32)
        pe[:,2*i] = torch.sin(args)
        pe[:,2*i+1] = torch.cos(args)
        self.register_buffer("PE", pe)

        self.linear = nn.Linear(d_model, vocab_size)

    # [BatchSize, seq_len, d_model]
    def forward(self, x_enc, x_dec):
        x_enc = x_enc + self.PE
        x_enc = self.encoder(x_enc)

        x_dec = x_dec + self.PE 
        x_dec = self.decoder(x_dec, x_enc)
        
        x = self.linear(x_dec)
        x = softmax(x, -1)
        return x
    
if __name__ == "__main__":

    B = 4
    seq_len = 768
    d_model = 512
    vocab_size = 37_000
    a = torch.rand([B, seq_len, d_model], device="mps")
    b = torch.rand([B, seq_len, d_model], device="mps")
    model = Transformer(vocab_size, seq_len)
    model.to("mps")
    o = model(a, b)
    loss = o.mean()
    loss.backward()