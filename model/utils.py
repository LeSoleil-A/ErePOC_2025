from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import torch
import torch.nn.functional as F
    
# Encoder-MLP Model Structure
class Net_embed_MLP(torch.nn.Module):
    # Two hidden layers
    def __init__(self, input_dim=1280, hidden_dim=512, out_dim=256, drop_prob=0.1):
        torch.nn.Module.__init__(self)
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(drop_prob),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(drop_prob),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        x = self.sequence(x)
        return F.normalize(x, dim=1)