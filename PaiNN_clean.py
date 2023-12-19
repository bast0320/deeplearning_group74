import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn import Linear

from message_clean import MessagePassPaiNN 
from update_clean import UpdatePaiNN
from dipole_clean import *

dictionary_targets = {
    0: 'Dipole moment',
    1: 'Isotropic polarizability',
    2: 'Highest occupied molecular orbital energy',
    3: 'Lowest unoccupied molecular orbital energy',
    4: 'Gap between HOMO and LUMO',
    5: 'Electronic spatial extent',
    6: 'Zero point vibrational energy',
    7: 'Internal energy at 0K',
    8: 'Internal energy at 298.15 K',
    9: 'Enthalpy at 298.15 K',
    10: 'Free energy at 298.15 K',
    11: 'Heat capacity at 298.15 K',
    12: 'Atomization energy at 0K',
    13: 'Atomization energy at 298.15 K',
    14: 'Atomization enthalpy at 298.15 K',
    15: 'Atomization free energy at 298.15 K',
    16: 'Rotational constant A',
    17: 'Rotational constant B',
    18: 'Rotational constant C'
}

class PaiNN(torch.nn.Module): 
    def __init__(
        self,
        num_feat,
        out_channels,
        num_nodes,
        cut_off=5.0,
        n_rbf=20,
        num_interactions=3,
        target=0
    ):
        super(PaiNN, self).__init__()
 
        self.num_interactions = num_interactions
        self.cut_off = cut_off
        self.n_rbf = n_rbf
        self.num_nodes = num_nodes
        self.num_feat = num_feat
        self.out_channels = out_channels
        self.lin = Linear(num_feat, num_feat)
        self.silu = Func.silu
        self.target = target
        

        self.list_message = nn.ModuleList(
            [
                MessagePassPaiNN(num_feat, out_channels, num_nodes, cut_off, n_rbf)
                for _ in range(self.num_interactions)
            ]
        )
        self.list_update = nn.ModuleList(
            [
                UpdatePaiNN(num_feat, out_channels, num_nodes)
                for _ in range(self.num_interactions)
            ]
        )
        
        # Gated Block network to calculate Dipole moment property
        if dictionary_targets[self.target] == 'Dipole moment':
            self.gate_net = mlp_gated_block(n_in = 128, 
                                            n_out = 1, 
                                            n_layers = 5, 
                                            n_hidden_nodes = 5)

    def forward(self, s, v, edge_index, edge_attr, r_ij):

        for i in range(self.num_interactions):
# Here we were very afraid of the blowing up, but in the end we did not need any batch normalization
            s_temp, v_temp = self.list_message[i](s, v, edge_index, edge_attr)
            s, v = s_temp + s, v_temp + v
            s_temp, v_temp = self.list_update[i](s, v)
            s, v = s_temp + s, v_temp + v

        if dictionary_targets[self.target] == 'Dipole moment':
            sl_and_vl = tuple((s, v))
            charges, atomic_dipoles = self.gate_net(sl_and_vl)
            scalar = calc_dipole(charges, atomic_dipoles, r_ij)
            
        else:
            s = self.lin(s)
            s = self.silu(s)
            s = self.lin(s)
            scalar = s

        return scalar

