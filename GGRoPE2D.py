# jerryxio.ng/posts/nd-rope
import math
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GoldenGateRoPE2d(nn.Module):
    def __init__(self, image_size, n_heads, n_freqs, min_freq=.8, max_freq=10, n_zero_freqs=0):
        super().__init__()
        intv = math.pi * (math.sqrt(5)-1)/2 # mod pi instead of 2pi # pi*(sqrt5+-1)/2 ; + and - are equivalent bec mod pi
        # intv = math.pi * (math.sqrt(5)-1) # https://en.wikipedia.org/wiki/Golden_angle
        speed = torch.cat([torch.zeros(n_zero_freqs), min_freq * (max_freq/min_freq) ** torch.linspace(0,1,n_freqs-n_zero_freqs)]) # [n_freqs]
        phi = torch.arange(n_heads*n_freqs).reshape(n_heads, n_freqs) * intv # [n_heads, n_freqs]
        direction = torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1) # [n_heads, n_freqs, 2]
        vel = speed.unsqueeze(-1) * direction # speed in direction[n_heads, n_freqs, 2]
        h, w = image_size
        xlim, ylim = math.sqrt(w/h), math.sqrt(h/w)
        y, x = torch.meshgrid(torch.linspace(-ylim, ylim, h), torch.linspace(-xlim, xlim, w), indexing="ij") # [h,w], y:row_num, x:col_num
        pos = torch.stack([x, y], dim=-1)[...,None,None,:] # [h,w,1,1,2] cartesian coords
        theta = (vel*pos).sum(dim=-1) # [h,w,n_heads,n_freqs,2]->[h,w,n_heads,d_head]
        cos, sin = torch.cos(theta), torch.sin(theta)
        self.affine = torch.stack([cos, -sin, sin, cos], dim=-1).unflatten(-1, (2,2))

    def forward(self, input): # [b,h,w,n_head,d_head]
        return (self.affine @ input.unflatten(-1, (-1,2)).unsqueeze(-1)).flatten(-3)

# image_size=(5,7)
# n_heads=4
# d_head=16
# ggrope = GoldenGateRoPE2d(image_size, n_heads, d_head//2)

# x = torch.rand(2, *image_size, n_heads, d_head)
# out = ggrope(x)
# print(out.shape)
