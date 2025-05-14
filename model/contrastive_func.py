import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrastive_loss:
    
    def __init__(self, margin=0.4):
        self.margin = margin
    
    def __call__(self, x, labels):
        """
        Calculate Contrastive Loss: KLDivergence --> 
        
        x      - n*m embedding matrix where n is batch size and m is embedding size
        labels - n Fingerprint arrays.
        """

        # calculate n*n cosine similarity matrix - Protein Similarity
        cos_sim = F.cosine_similarity(x, x.unsqueeze(1), dim=2)
        
        ##################### Calculate the ligand similarity matrix. #####################
        lig_sim = torch.matmul(labels, labels.T) / (torch.sum(labels,axis=1).view(-1, 1) + torch.sum(labels, axis=1).view(1, -1).repeat(labels.size(0), 1) - torch.matmul(labels, labels.T))
        lig_sim = lig_sim.cuda()

        ##################### Remove diagonal and flatten #####################
        # remove diagonal values (self cosine equal to 1) #
        cos_sim_no_diag = torch.masked_select(cos_sim, ~torch.eye(cos_sim.size(0), device=cos_sim.device).bool())
        # vector equivalent to 'cos_sim_no_diag' of same label or different label --> remove the diagonal
        lig_sim_no_diag = torch.masked_select(lig_sim, ~torch.eye(lig_sim.size(0), device=lig_sim.device).bool())
        # add one dimension
        cos_sim_no_diag = torch.unsqueeze(cos_sim_no_diag,0)
        lig_sim_no_diag = torch.unsqueeze(lig_sim_no_diag,0)
        ## Calculate loss
        cos_sim_no_diag_log = F.log_softmax(cos_sim_no_diag, dim=1)
        lig_sim_no_diag_tar = F.softmax(lig_sim_no_diag, dim=1)
        kl = nn.KLDivLoss(reduction='batchmean')
        loss = kl(cos_sim_no_diag_log, lig_sim_no_diag_tar)
        
        return loss