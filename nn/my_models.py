from textwrap import indent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import numpy as np

from PIL import Image

class seq_model(nn.Module):
    def __init__(self, ind, outd, w):
        super(seq_model, self).__init__()
        self.ind=ind
        self.outd=outd
        self.w=w
        self.conv=nn.Conv1d(ind, outd, kernel_size=self.w)
        
    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]

        x = x.permute(0,2,1)
        seqFt = self.conv(x)
        seqFt = torch.mean(seqFt,-1)

        return seqFt

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)
