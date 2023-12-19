import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.data import Batch, Data, DataLoader
from torch_geometric.nn import radius, radius_graph


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        
        self.cutoff = cutoff

    def forward(self, distances):
        
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class BesselBasis(torch.nn.Module):
    
    def __init__(self, cutoff=5.0, n_rbf=None):
       
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        inputs = torch.norm(inputs, p=2, dim=1)
        a = self.freqs
        ax = torch.outer(inputs, a)
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[:, None]

        return y


class Bipartite(Data):
    def __init__(self, edge_index, coord_elec, coord_nuc, s_nuc, v_nuc, num_nodes):
        super(Bipartite, self).__init__()
        self.edge_index = edge_index
        self.coord_elec = coord_elec
        self.coord_nuc = coord_nuc
        self.s_nuc = s_nuc
        self.v_nuc = v_nuc
        self.num_nodes = num_nodes

    def __inc__(self, key, value):
        if key == "edge_index":
            return torch.tensor([[self.coord_nuc.size(0)], [self.coord_elec.size(0)]])
        else:
            return super().__inc__(key, value)
