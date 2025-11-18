# @title rope
import math
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RoPE1(nn.Module):
    def __init__(self, dim, seq_len=512, top=torch.pi, base=1000):
        super().__init__()
        speed = top / (base ** (torch.arange(0, dim, step=2) / dim)) # [dim//2]
        pos = torch.arange(seq_len).unsqueeze(-1) # [t,1]
        theta = (speed*pos) # [t,1]*[dim//2]=[t,d//2]
        self.theta = theta
        cos, sin = torch.cos(theta), torch.sin(theta)
        self.affine = torch.stack([cos, -sin, sin, cos], dim=-1).unflatten(-1,(2,2)).to(device) # [t,d//2,4]-> [1,1,t,d//2,2,2]

    def forward(self, x): # [b,h,t,d]
        return (self.affine @ x.unflatten(-1, (-1,2)).unsqueeze(-1)).flatten(-3) # [1,1,t,d//2,2,2] @ [b,h,t,d//2,2,1] = [b,h,t,d]


class RoPE2(nn.Module):
    def __init__(self, dim, seq_len=512, min_freq=1, max_freq=400, n_zero_freqs=0):
        super().__init__()
        speed = torch.cat([torch.zeros(n_zero_freqs), min_freq * (max_freq/min_freq) ** torch.linspace(0,1,dim//2-n_zero_freqs)]) # [dim//2]
        pos = torch.linspace(0, 1, seq_len).unsqueeze(-1) # [t,1]
        theta = (speed*pos) # [t,1]*[dim//2]=[t,d//2]
        self.theta = theta
        cos, sin = torch.cos(theta), torch.sin(theta)
        self.affine = torch.stack([cos, -sin, sin, cos], dim=-1).unflatten(-1,(2,2)).to(device) # [t,d//2,4]-> [1,1,t,d//2,2,2]

    def forward(self, x): # [b,h,t,d]
        return (self.affine @ x.unflatten(-1, (-1,2)).unsqueeze(-1)).flatten(-3) # [1,1,t,d//2,2,2] @ [b,h,t,d//2,2,1] = [b,h,t,d]

# dim=64
# seq_len=64
# rope2 = RoPE1(dim, seq_len, top=torch.pi, base=100)
# # rope2 = RoPE2(dim, seq_len, min_freq=1, max_freq=200, n_zero_freqs=0)

# x = torch.rand(2, n_heads, seq_len, dim, device=device)
# out = rope2(x)
# print(out.shape)

# theta = rope2.theta # [t,d//2]
# sim = torch.cos(theta-theta[0].unsqueeze(0)).T
# # print(sim.shape)

# import numpy as np
# import matplotlib.pyplot as plt
# def imshow(img):
#     npimg = img.numpy()
#     print(npimg.shape)
#     plt.rcParams["figure.figsize"] = (8,8)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# import torchvision
# imshow(torchvision.utils.make_grid(sim, nrow=n_freqs))
