import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class GatedEquivariantBlock(torch.nn.Module):
    """
    Builds the gated block following the steps in figure 3.

    Args:
        features_in: number of input features.
        features_out: number of output features.
        n_hidden: number of hidden nodes
        sl_and_vl: tuple contatining the layered s and v
    """

    def __init__(self, features_in: int, features_out: int, n_hidden: int):
        super(GatedEquivariantBlock, self).__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.n_hidden = n_hidden
        
        # Define parameters and layers
        self.W_left = nn.Linear(features_in, features_out, bias = False)
        self.W_right = nn.Linear(features_in, features_out, bias = False)
        self.W1 = nn.Linear(features_in+features_out, n_hidden, bias = True)
        self.W2 = nn.Linear(n_hidden, features_out+features_out*3, bias = True)

    def forward(self, 
                sl_and_vl: tuple):
        
        # Unpack tuple and dimensionality handling
        sl, vl = sl_and_vl
        v = vl.flatten(-2)
        flatv = v.shape[-1]
        v_correct_dim = v.reshape(-1, int(flatv/3), 3)
        v_correct_dim_t = torch.transpose(v_correct_dim, 1, 2)
        v_correct_dim_t = v_correct_dim_t.to(dtype=torch.float32)
        
        # W * 
        outW_left = self.W_left(v_correct_dim_t)
        outW_right = self.W_right(v_correct_dim_t)
        
        # Further calculations of the right-path:
        normW = torch.norm(outW_right, dim = 1)
        stack = torch.cat([sl, normW], dim = -1)
        
        # Scalar network (W * + b)
        outW1 = self.W1(stack)
        outW1_silu = F.silu(outW1)  #activation
        a = self.W2(outW1_silu)

        # Split
        split_sl, split_vl = torch.split(a, [self.features_out, self.features_out*3], dim=1)
        split_vl = split_vl.reshape(-1, 3, self.features_out)
        delta_sl_update = split_sl
        
        # Multiply element-wise and correct dimensions back
        delta_vl_update = torch.mul(split_vl, outW_left)
        delta_vl_update = torch.transpose(delta_vl_update, 1, 2)
        
        # Packing back into a tuple
        dsl_and_dvl = tuple((delta_sl_update, delta_vl_update))
        
        return dsl_and_dvl
    

def mlp_gated_block(n_in: int,
                    n_out: int,
                    n_layers: int,
                    n_hidden_nodes: int) -> nn.Module:
    """
    Build neural network analog to MLP with `GatedEquivariantBlock`s.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_layers: number of layers.
        n_hidden_nodes: number of hidden nodes within GatedEquivariantBlock
    """
    # List of input and output nodes in each layer: (halfes each layers, but does not go below n_out)
    features_in = [n_in] * n_layers
    features_out = [n_out] * n_layers

    for i in range(n_layers-1):
        n_hidden_notes = round((features_in[i]+n_out) / 2)
        features_in[i+1] = n_hidden_notes
        features_out[i]  = n_hidden_notes

    # Running GatedEquivariantBlock for each layer
    layers = list()
    for layer in range(n_layers):
        layers.append(GatedEquivariantBlock(features_in = features_in[layer], 
                                            features_out = features_out[layer], 
                                            n_hidden = n_hidden_nodes))

    # Collecting the layers and their weights to make the network
    gate_net = nn.Sequential(*layers)
    return gate_net



def calc_dipole(charges,
                atomic_dipoles,
                r_ij) -> nn.Module:
    """
    Caculates the Tensorial property dipole moment
    """
    
    # Dimensionality handling...        
    positions = r_ij.unsqueeze(1)
    charges = charges.unsqueeze(2)

    # Main calculation (equation 13) 
    y = atomic_dipoles + charges * positions

    # sum to atomic values (comparable with PaiNN function)
    atomic_sum = torch.sum(y, dim = 1)

    return atomic_sum

    