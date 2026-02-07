# @title sincos_2d
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class sinusodial(nn.Module): # Rotary Positional Embeddings, flexible pos
    def __init__(self, dim, top=torch.pi, base=1000):
        super().__init__()
        self.theta = top / (base ** (torch.arange(0, dim, step=2, device=device) / dim))
        # self.theta = top / (base ** torch.linspace(0, 1, dim//2, device=device))

    def forward(self, pos): # [batch] in [0,1]
        angles = (pos.unsqueeze(-1) * self.theta).unsqueeze(-1) # [seq_len, 1] * [dim // 2] -> [seq_len, dim // 2, 1]
        rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # [seq_len, dim // 2, 2]
        return rot_emb.flatten(-2) # [seq_len, dim]


def sinusodial(dim, seq_len=512, top=torch.pi, base=1000):
    theta = top / (base ** (torch.arange(0, dim, step=2) / dim))
    pos = torch.arange(seq_len).unsqueeze(-1)
    angles = (pos * theta)[None,...,None] # [seq_len, 1] * [dim//2] -> [1, seq_len, dim//2, 1]
    rot_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2).to(device) # [1, seq_len, dim//2, 2] -> [1, seq_len, dim]
    return rot_emb


# https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
def sincos_2d(embed_dim, hw):
    h,w = hw
    grid = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device)), axis=0)
    print(grid[:1].shape, grid[1:].shape)
    emb_h = sincos_1d(embed_dim//2, grid[:1]) # (H*W, D/2)
    emb_w = sincos_1d(embed_dim//2, grid[1:]) # (H*W, D/2)
    pos_embed = torch.cat([emb_h, emb_w], axis=1)  # (H*W, D)
    return pos_embed

def sincos_1d(embed_dim, grid, base=10000):
    # omega = 1/base**(torch.arange(embed_dim//2)*2/embed_dim)
    omega = 1/base**(torch.linspace(0,1, embed_dim//2, device=device))
    # print(omega)
    out = torch.einsum('m,d->md', grid.flatten(), omega)   # (M, D/2), outer product
    pos_embed = torch.cat([torch.sin(out), torch.cos(out)], axis=1)  # (M, D)
    return pos_embed # [M, D]

# emb = sincos_2d(64, 6)
emb = sincos_2d(64, (4,6))
print(emb.shape)
print(emb)

