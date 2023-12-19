import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch_geometric.nn import MessagePassing

from helper_clean import BesselBasis, CosineCutoff

# Here we use the message passing from torch geometric as it is faster. Another version is also attached where we have done this ourselves.
class MessagePassPaiNN(MessagePassing):
    def __init__(self, num_feat, out_channels, num_nodes, cut_off=5.0, n_rbf=20):
        super(MessagePassPaiNN, self).__init__(aggr="add")

        self.lin1 = nn.Linear(num_feat, out_channels)
        self.lin2 = nn.Linear(out_channels, 3 * out_channels)
        self.lin_rbf = nn.Linear(n_rbf, 3 * out_channels)
        self.silu = Func.silu

        self.RBF = BesselBasis(cut_off, n_rbf)
        self.f_cut = CosineCutoff(cut_off)
        self.num_nodes = num_nodes
        self.num_feat = num_feat

    def forward(self, s, v, edge_index, edge_attr):

        s = s.flatten(-1)
        v = v.flatten(-2)

        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]

        x = torch.cat([s, v], dim=-1)

        x = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            flat_shape_s=flat_shape_s,
            flat_shape_v=flat_shape_v,
        )

        return x

    def message(self, x_j, edge_attr, flat_shape_s, flat_shape_v):

        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)

        # r_ij channel
        rbf = self.RBF(edge_attr)
        ch1 = self.lin_rbf(rbf)
        cut = self.f_cut(edge_attr.norm(dim=-1))
        W = torch.einsum("ij,i->ij", ch1, cut)  # ch1 * f_cut

        # s_j channel
        phi = self.lin1(s_j)
        phi = self.silu(phi)
        phi = self.lin2(phi)

        # Split

        left, middle, right = torch.split(phi * W, self.num_feat, dim=-1)

        # v_j channel
        normalized = Func.normalize(edge_attr, p=2, dim=1)
        v_j = v_j.reshape(-1, int(flat_shape_v / 3), 3)
        hadamard_right = torch.einsum("ij,ik->ijk", right, normalized)
        hadamard_left = torch.einsum("ijk,ij->ijk", v_j, left)
        dvm = hadamard_left + hadamard_right

        # Prepare vector for update
        x_j = torch.cat((middle, dvm.flatten(-2)), dim=-1)

        return x_j

    def update(self, out_aggr, flat_shape_s, flat_shape_v):

        s_j, v_j = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)

        return s_j, v_j.reshape(-1, int(flat_shape_v / 3), 3)

