
if __name__ == '__main__':
    from pyinstrument import Profiler
    with Profiler(interval=0.01) as profiler:
        from schnetpack.datasets import QM9
        import torch as torch
        from schnetpack.transform import ASENeighborList
        import matplotlib.pyplot as plt
        import torch.optim as optim
        import os
        from torch_geometric.nn import radius, radius_graph
        # from torch_geometric.data import Batch, Data, DataLoader
        import torch_cluster
        import torch
        import torch.nn as nn
        from torch.nn import Linear
        import numpy as np
        import pandas as pd

        import torch.nn.functional as F
        import math
        from torchcontrib.optim import SWA
        import torch_cluster
        from torch_geometric.nn import radius, radius_graph
        # from torch_geometric.data import Batch, Data, DataLoader
        import torch
        import torch.nn as nn
        from torch.nn import Linear
        from collections import defaultdict
        from itertools import chain
        from torch.optim import Optimizer
        import torch
        import warnings

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        # from IPython.display import clear_output
        from IPython import display

        import numpy as np

        import torch
        import numpy as np
        import pytorch_lightning as pl
        from torch_geometric.data import Data
        from torch_geometric.datasets import QM9
        from typing import Optional, List, Union
        from torch_geometric.loader import DataLoader
        from torch_geometric.transforms import BaseTransform


        # class GetTarget(BaseTransform):
        #     def __init__(self, target: Optional[int] = None) -> None:
        #         self.target = [target]
            
        #     def forward(self, data: Data) -> Data:
        #         if self.target is not None:
        #             data.y = data.y[:, self.target]
        #         return data

        class GetTarget(BaseTransform):
            def __init__(self, target: Optional[int] = None, mean: float = 0.0, std: float = 1.0) -> None:
                self.target = [target]
                self.mean = mean
                self.std = std

            def forward(self, data: Data) -> Data:
                if self.target is not None:
                    data.y = (data.y[:, self.target] - self.mean) / self.std
                return data
    
        class QM9DataModule(pl.LightningDataModule):

            target_types = ['atomwise' for _ in range(19)]
            target_types[0] = 'dipole_moment'
            target_types[5] = 'electronic_spatial_extent'

            # Specify unit conversions (eV to meV).
            unit_conversion = {
                i: (lambda t: 1000*t) if i not in [0, 1, 5, 11, 16, 17, 18]
                else (lambda t: t)
                for i in range(19)
            }

            def __init__(
                self,
                target: int = 0,
                data_dir: str = 'data/',
                batch_size_train: int = 32,
                batch_size_inference: int = 32,
                num_workers: int = 0,
                splits: Union[List[int], List[float]] = [0.8, 0.1, 0.1],
                seed: int = 0,
                subset_size: Optional[int] = None,
            ) -> None:
                super().__init__()
                self.target = target
                self.data_dir = data_dir
                self.batch_size_train = batch_size_train
                self.batch_size_inference = batch_size_inference
                self.num_workers = num_workers
                self.splits = splits
                self.seed = seed
                self.subset_size = subset_size

                self.data_train = None
                self.data_val = None
                self.data_test = None


            def prepare_data(self) -> None:
                # Download data
                QM9(root=self.data_dir)


            def setup(self, stage: Optional[str] = None) -> None:
                dataset = QM9(root=self.data_dir, transform=GetTarget(self.target))

                # Shuffle dataset
                rng = np.random.default_rng(seed=self.seed)
                dataset = dataset[rng.permutation(len(dataset))]

                # Subset dataset
                if self.subset_size is not None:
                    dataset = dataset[:self.subset_size]
                
                # Split dataset
                if all([type(split) == int for split in self.splits]):
                    split_sizes = self.splits
                elif all([type(split) == float for split in self.splits]):
                    split_sizes = [int(len(dataset) * prop) for prop in self.splits]

                split_idx = np.cumsum(split_sizes)
                self.data_train = dataset[:split_idx[0]]
                self.data_val = dataset[split_idx[0]:split_idx[1]]
                self.data_test = dataset[split_idx[1]:]
    # STANDARDIZING...
    # Note that you should only use the statistics from the training set to avoid data leakage.
                # Calculate mean and std of target variable in training set
                # extract the targets from the training set
                # print("Input features example: ", self.data_train[0].pos)
                # target_values = [self.data_train[k].y.item() for k in range(len(self.data_train))]
                # # target_values = [data.y[:, self.target].item() for data in self.data_train]
                # mean = np.mean(target_values)
                # std = np.std(target_values)

                # # Standardize target variable in all datasets
                # transform = GetTarget(self.target, mean, std)
                # self.data_train = QM9(root=self.data_dir, transform=transform)
                # self.data_val = QM9(root=self.data_dir, transform=transform)
                # self.data_test = QM9(root=self.data_dir, transform=transform)


            def get_target_stats(self, remove_atom_refs=False, divide_by_atoms=False):
                atom_refs = self.data_train.atomref(self.target)

                ys = list()
                for batch in self.train_dataloader(shuffle=False):
                    y = batch.y.clone()
                    if remove_atom_refs and atom_refs is not None:
                        y.index_add_(
                            dim=0, index=batch.batch, source=-atom_refs[batch.z]
                        )
                    if divide_by_atoms:
                        _, num_atoms  = torch.unique(batch.batch, return_counts=True)
                        y = y / num_atoms.unsqueeze(-1)
                    ys.append(y)

                y = torch.cat(ys, dim=0)
                return y.mean(), y.std(), atom_refs


            def train_dataloader(self, shuffle=True) -> DataLoader:
                return DataLoader(
                    self.data_train,
                    batch_size=self.batch_size_train,
                    num_workers=self.num_workers,
                    shuffle=shuffle,
                    pin_memory=True,
                )


            def val_dataloader(self) -> DataLoader:
                return DataLoader(
                    self.data_val,
                    batch_size=self.batch_size_inference,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=True,
                )


            def test_dataloader(self) -> DataLoader:
                return DataLoader(
                    self.data_test,
                    batch_size=self.batch_size_inference,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=True,
                )
        

        # OLD DATA set loader.... 
        # os.remove('split.npz')
        # qm9data = QM9(
        #     './qm9.db', 
        #     batch_size=10,
        #     num_train=110000,
        #     num_val=10000,
        #     transforms=[ASENeighborList(cutoff=5.)]
        # )
        # qm9data.prepare_data()
        # qm9data.setup()

        # import pickle
        # with open('qm9data_w_adj.pickle', 'rb') as handle:
        #     qm9data = pickle.load(handle)


        def create_adjacency_matrix(coordinates, cutoff):
            # Calculate pairwise squared distances using broadcasting
            diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            
            # Calculate Euclidean distances
            distances = np.sqrt(sq_dist)
            
            # Apply the cutoff to populate the adjacency matrix
            adjacency_matrix = np.where(distances < cutoff, 1, 0)
            
            return adjacency_matrix

        #Creates direction list for each atom in one molecule
        def create_direction_matrix(coordinates,cutoff):
            # Calculate pairwise squared distances using broadcasting
            diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]

            sq_dist = np.sum(diff ** 2, axis=-1)
            
            # Calculate Euclidean distances
            distances = np.sqrt(sq_dist)
            
            r_ij= [ [] for _ in range(0,np.shape(coordinates)[0]) ]

            for i in range(0,np.shape(coordinates)[0]):
                for j in range(0,np.shape(coordinates)[0]):
                    if np.sum(distances[i,j])<cutoff and np.sum(distances[i,j])>0:  ##Added cutoff
                        r_ij[i].append((diff[i,j]))
                    else:
                        r_ij[i].append([])

            return r_ij


        #Creates directions for each atom in each molecule
        def create_r(start = 0, end = 100, cutoff = 1.2):
            r2_ij= [ [] for _ in range(0, end-start) ]
            r3_ij= [ [] for _ in range(0, end-start) ]

            for i in range(start,end): #Molecule
                J1 = qm9data.dataset[i]["_n_atoms"].item()
                for j in range(0,J1): #Atom in molecule
                    k1 = create_direction_matrix(np.array(qm9data.dataset[i]["_positions"]),cutoff)[j]

                    #Rewrite to list of tensors
                    filtered_arrays = [arr for arr in k1 if len(arr) > 0]
                    tens = np.array(filtered_arrays)
                    tens = torch.from_numpy(tens)
                    r2_ij[i-start].append(tens.to(dtype=torch.float32))

            for i in range(0,end-start):
                r3_ij[i]=torch.cat(r2_ij[i], dim=0)

            return r3_ij

        # We will here create the embedding s_i from the Z_i, which is the atom numbers
        # This embedding will for each molecule produce a 128 dimensional vector, s_i


        max_atom_number = 30

        # This is the green block from fig 2.a: 
        class AtomEmbedding(nn.Module):
            def __init__(self, num_embeddings, embedding_dim):
                super(AtomEmbedding, self).__init__()
                self.embedding = nn.Embedding(num_embeddings, embedding_dim) # main line

            def forward(self, atomic_numbers):
                return self.embedding(atomic_numbers)

        # Example usage: here we initialize an instance of the class, which can be used to embed atom numbers as a function

        atom_embedding = AtomEmbedding(num_embeddings=max_atom_number, embedding_dim=128)


        # for each atom i in the molecule, we will embed the atom numbers
        # we will start with doing this for the first molecule in the dataset

        # embeddings = atom_embedding(qm9data.dataset[0]["_atomic_numbers"])

        # General framework, not done!
        device = torch.device("mps")
        def radial_basis_function(r_ij, r_cut, n=20):
            """
            Apply a radial basis function to a tensor of vectors r_ij, only for vectors with magnitude <= r_cut.
            r_cut: The cutoff radius.
            n: The order of the basis function.
            """

            n_array = torch.tensor(np.array(range(1,n+1)), device=device, dtype=torch.float32)
            # Calculate the norm of each vector
            r_norm = torch.norm((r_ij), p = 2, dim = 1)
            
            # Create a mask for vectors within the cutoff radius
            # mask = r_norm <= r_cut
            rbf = []
            for norm in r_norm:
                if norm <= r_cut:
                    rbf.append(torch.sin( n_array * torch.pi * norm / r_cut) / norm)

            # Calculate the radial basis function, apply mask to zero out values beyond r_cut
            # rbf = torch.sin( n_array * torch.pi * r_norm / r_cut) / r_norm
            # rbf = torch.where(mask, rbf, torch.zeros_like(rbf))

            rbf = torch.cat(rbf).reshape(-1, 20)
            
            return rbf

        r_ij = torch.tensor([[1, 2, 3], [4, 5, 6], [11, 12, 13]], dtype=torch.float32)
        r_ij_norm = torch.norm(r_ij, p = 2, dim = 1)
        # import matplotlib.pyplot as plt
        # plt.plot(np.array(radial_basis_function(r_ij, r_cut = 25, n = 20).cpu()).transpose())



        def cutoff_cosine(input, cutoff):
            # Compute values of cutoff function
            f_cut = 0.5 * (1+torch.cos(input * math.pi / cutoff))
            # Remove contributions beyond the cutoff radius
            f_cut *= (input < cutoff).float()
            return f_cut




        class MessageFunction(nn.Module):
            def __init__(self, s_dim, n_atoms, out_dim = 128, r_cut_param = 5):
                super(MessageFunction, self).__init__()
                
                self.W1 = nn.Linear(s_dim, out_dim, bias = True) # see picture above to see that W1 is top W.+b block
                self.W2 = nn.Linear(20, 3*out_dim, bias = True) 
                self.W3 = nn.Linear(s_dim, 3*out_dim, bias = True)
                self.r_cut_param = r_cut_param
                self.bn_message_s = nn.BatchNorm1d(out_dim)
                self.bn_message_v = nn.BatchNorm1d(out_dim)

            
            def forward(self, s, v, r_ij , edge_index): #, edge_attr ):
                # print("s, v, r_ij shapes: ", s.shape, v.shape, r_ij.shape)
                # Dimensionality handling...
                s = s.flatten(-1)
                v = v.flatten(-2)
                flatv = v.shape[-1]
                # print("flatv: ", flatv)
                flats = s.shape[-1]
                v_correct_dim = v.reshape(-1, int(flatv/3), 3)
                v_correct_dim_t = torch.transpose(v_correct_dim, 1, 2) # transpose the last two dimensions, so we can sum by the feature dimension (I'm pretty sure)
                # print("v_correct_dim_t shape: ", v_correct_dim_t.shape)
                r_normalized = r_ij/torch.norm((r_ij))
                rbf = radial_basis_function(r_ij, r_cut = self.r_cut_param, n = 20) # n is fixed here, but could be parametrized

                # print("shape of r_ij", r_ij.shape)
                phi = self.W3(F.silu( self.W1(s)) ) # this is the phi function from the paper

                scriptW = self.W2(rbf) 
                # print(scriptW.shape)
                scriptW = cutoff_cosine(scriptW, cutoff = 4.5)
                # print(scriptW.shape)
                # print("phi shape: ", phi.shape)
                # print("scriptW shape: ", scriptW.shape)
                split = phi*scriptW
                # print("split: ", split)
                # print("split shape: ", split.shape)
                split1, split2, split3 = torch.split(split, 128, dim = -1)
                
                #print("split3 og r_norm. shape: ", split3.shape, r_normalized.shape)

                #print("shape of einsum: ", torch.einsum('ij,ik->jk', split3, r_normalized).shape)

                sp3_r_norm = torch.einsum('ij,ik->ijk', split3, r_normalized) # split3 * r_normalized

                # print("split1, v_correct_dim_t og sp3_r_norm shape: ", split1.shape, v_correct_dim_t.shape, sp3_r_norm.shape)
                
                s1_v = torch.einsum('ijk,ik->ikj', v_correct_dim_t, split1 ) # split1*v_correct_dim_t
                # print("s1_v shape: ", s1_v.shape)
                
                delta_v = s1_v + sp3_r_norm

                ## OUTCOMMENTED....
                # print("shape delta_v: ", delta_v.shape)
                # delta_v sum over j, WHAT DIM IS THAT ...--
                # delta_v_message = torch.sum(delta_v, dim = 0)
                # print("delta_v_message shape: ", delta_v_message.reshape(-1, int(flatv / 3), 3).shape)
                # delta_s_message = torch.sum(split2, dim = 0)

                # I want to obv. return an s vector that is as big as the one we got in, but for the i'th entry, we only want to sum these
                # j's that is connected to i.
                # split2 is [~9000, 128] and we want to sum over the 9000, but only for the j's that is connected to i.
                # We have the edge_index, which is a list of tuples, where each tuple is a connection between two atoms.
                
                # based on edge_index_batch we sum the split2 over the first dimension
                delta_s_message = torch.zeros_like(s, device=device, dtype=torch.float32)
                delta_v_message = torch.zeros_like(delta_v, device=device, dtype=torch.float32)
                for j in range(len(edge_index[0,:])):
                    delta_s_message[edge_index[1,j],:] += split2[edge_index[0,j],:]
                    delta_v_message[edge_index[1,j],:,:] += delta_v[edge_index[0,j],:,:]


                print("delta s message: ", delta_s_message.shape)
                
                # delta_v_message.index_add_(0, edge_index_batch[0,:], delta_v)

                # print("delta_s_message shape: ", delta_s_message.shape)
                # print("delta_v_message shape: ", delta_v_message.shape)
                # print("split2 shape: ", split2.shape) 
                return delta_s_message, delta_v_message
            # Returns a tensor of messages


        class GatedEquivariantBlock(nn.Module):
            def __init__(self,  s_dim, v_dim, n_atoms, out_dim, out_dimv, n_hidden):
                super(GatedEquivariantBlock, self).__init__()
                self.s_dim = s_dim
                self.v_dim = v_dim
                self.n_atoms = n_atoms
                self.out_dim = out_dim
                self.out_dimv = out_dimv
                self.n_hidden = n_hidden
                
                # Linear layer for mixing of v
                self.mix_v = nn.Linear(v_dim, 2*out_dimv, bias = False)
                
                # Define layers and parameters for the update function
                self.W1 = nn.Linear(s_dim+out_dimv, n_hidden, bias = True)
                self.W2 = nn.Linear(n_hidden, s_dim+out_dimv, bias = True)

            def forward(self, sl, vl):
                
                # Vector mixing
                vl_mix = self.mix_v(vl)
                
                # The 2 W-paths:
                W_right, W_left = torch.split(vl_mix, self.out_dimv, dim=-1)
                normW_right = torch.norm(W_right, dim = -2)
                stack = torch.cat([sl, normW_right], dim = -1)
                
                # Scalar network
                outW1 = self.W1(stack)
                outW1_silu = F.silu(outW1)
                a = self.W2(outW1_silu)
                
                # Vector layer+1
                d_sl, a = torch.split(a, [self.s_dim, self.out_dimv], dim=-1)
                d_vl = a.unsqueeze(-2) * W_left
                return d_sl, d_vl
                
        ### Testing gated block ###
        n_atoms = 5*10
        s_dim = 128
        out_dim = 128
        s0 = torch.rand(n_atoms, out_dim, dtype=torch.float)
        v0 = torch.zeros(n_atoms, out_dim, 3, dtype=torch.float)
        # MyBlock = GatedEquivariantBlock(s_dim = 128,
        #                                 v_dim = 3,
        #                                 n_a5050,
        #                                 out_dimv = 3,
        #           n_atoms       ,              out_dim = 128,
        #                                 n_hidden = 5)
        # MyBlockOutput = MyBlock(s0, v0)
        # d_sl, d_vl = MyBlockOutput

        # d_vl.shape


        ## Draft for dipole calculation 
        # omitted charge correction 
        # only calculations for vector representation 

        class TensorialProperties(nn.Module):
            def __init__(self,  s_dim, v_dim, n_atoms, out_dim, out_dimv, idx_mol, n_hidden, r_i):
                super(GatedEquivariantBlock, self).__init__()
                self.s_dim = s_dim
                self.v_dim = v_dim
                self.n_atoms = n_atoms
                self.out_dim = out_dim
                self.out_dimv = out_dimv
                self.n_hidden = n_hidden
                self.idx_mol = idx_mol
                self.positions = r_i_test     #FIX THIS 

            
            def calc_dipole(self, d_sl, d_vl):

                max_mol = int(self.idx_mol[-1]) + 1
                
                # use output of gated block
                charges = d_sl
                atomic_dipoles = torch.squeeze(d_vl , -1)

                # calc. dipole using eqn. 13
                y = atomic_dipoles + (self.positions * charges) #need to de ##### *-ISSUEfine positions 
                print(y)
                
                # sum over all atoms in molecule(?)
                #dipoles = scatter_add(y, self.idx_mol, dim_size=max_mol) #sum over atoms - use scatter_add

                dipoles = [0] * (max_mol+1)
                for i in range(max_mol+1):      
                    index = idx_mol==i          #idx_mol must be numpy
                    dipoles[i] = sum(y[index])  #y must be numpy

                return dipoles
            
                print(dipoles.shape())
            


            #def polarizability(self)





        class UpdateFunction(nn.Module):
            def __init__(self, s_dim, n_atoms, out_dim = 128):
                super(UpdateFunction, self).__init__()

                # Define layers and parameters for the update function
                self.U = nn.Linear(s_dim, out_dim, bias = False)
                self.V = nn.Linear(s_dim, out_dim, bias = False)
                self.W1 = nn.Linear(s_dim*2, out_dim, bias = True) # see picture above to see that W1 is top W.+b block
                self.W2 = nn.Linear(s_dim, 3*out_dim, bias = True) 
                
                # We introduce batch normalization here...
                self.bn_update_s = nn.BatchNorm1d(out_dim)
                self.bn_update_v = nn.BatchNorm1d(out_dim)
            
            def forward(self, s, v):
                
                # Dimensionality handling...
                # print("v shape start: ", v.shape )
                # print("s shape start: ", s.shape )
                # v = self.bn_update_v(v)
                # s = self.bn_update_s(s)
                # print("v shape AFTER BN: ", v.shape )
                s = s.flatten(-1)
                v = v.flatten(-2)
                # print(" v shape AFTER FLATTEN: ", v.shape )
                # print("s BF BN:          ", s[0:5, 0:5])
                # s = self.bn_update_s(s)
                # print("s AFTER BN: -----      ", s[0:5, 0:5])
                # print("s shape", s.shape)

                # print("v: ", v) # v is ok...
                flatv = v.shape[-1]
                flats = s.shape[-1]
                # print("flatv: ", flatv)

                # print("v shape: ", v[0:3,0:9])
                v_correct_dim = v.reshape(-1, int(flatv/3), 3)
                # print("v_correct_dim: ", v_correct_dim.shape)
                v_correct_dim_t = torch.transpose(v_correct_dim, 1, 2) # transpose the last two dimensions, so we can sum by the feature dimension (I'm pretty sure)
                
                # print("v type: ", type(v_correct_dim_t))
                v_correct_dim_t = v_correct_dim_t.to(dtype=torch.float32)
                # print("v after trans: ", v_correct_dim_t.shape)
                # print("v shape: ", v.shape)
                # print("v_correct_dim_t: ", v_correct_dim_t) # very big numbers here....
                # Main calcs begin...
                
                outV = self.V(v_correct_dim_t)
                # print("outV : ", outV[0:5,0:3]) # some very big numbers here 
                outU = self.U(v_correct_dim_t)

                # print("outV shape: ", outV.shape)
                normV = torch.norm(outV, dim =1)

                # print("normV shape: ", normV.shape)
                # normV should be of length 128, so we can concatenate it with s
                # we are taking the norm over each of the 3 dim

                stack = torch.cat((s, normV), dim = -1)
                
                # print("stack : ", stack) some infinities here

                outW1 = self.W1(stack)
                outW1_silu = F.silu(outW1)
                a = self.W2(outW1_silu)
                # SPLIT a....
                a_vv, a_sv, a_ss = torch.split(a, 128, dim=1)
            
                # a_sv = a[:, 128:256]
                # a_ss = a[:, 256:]

                # print("outU and outV shape", outU.shape, outV.shape)

                outV_dot = torch.einsum("ijk, ijk->ik", outU, outV)

                # print("outV_dot: ", outV_dot.shape) # nans
                
                # print("a_sv shape", a_sv.shape)
                # print("outV_dot shape: ", outV_dot.shape) 
                # print("a_vv og outU shape: ", a_vv.shape, outU.shape)
                delta_v_update = torch.einsum("ij,ikj->ijk", a_vv, outU)
                # print("delta_v_update shape: ", delta_v_update.shape)
                delta_s_update = (a_sv*outV_dot) + a_ss

                # print(delta_v_update, delta_s_update) # here is nan....
                print("v shape end: ", delta_v_update.reshape(-1, int(flatv / 3), 3).shape )
                # print("new iteration ---------------------------------------------")
                # print("delta_s_update shape: ", delta_s_update.shape)
                
                return delta_s_update , delta_v_update.reshape(-1, int(flatv / 3), 3) 



        def sum_into_tensor(values, indices):
            """
            Sums the numbers in 'values' into a tensor at the positions specified by 'indices'.
            
            Parameters:
            values (list): A list of numbers to sum.
            indices (list): A list of indices where the sums should be placed.
            
            Returns:
            np.ndarray: A tensor with the sums placed at the specified indices.
            """
            
            # Find the maximum index to determine the size of the tensor
            num_classes = max(indices)
            
            # Initialize a tensor of zeros with the size equal to the number of classes
            tensor = np.zeros(num_classes)
            
            # Iterate over the values and their corresponding indices
            for value, idx in zip(values, indices):
                tensor[idx-1] += value  # Subtract 1 from idx because numpy arrays are 0-indexed
            
            return tensor



        # Putting it all together fig2.a

        class PaiNN(torch.nn.Module):
            def __init__(
                self,
                s_dim,
                out_dim,
                n_atoms,
                cut_off=5,
                num_interactions=3,  # standard in the paper, but could be modified for generality :)
                ):
                super(PaiNN, self).__init__()

                self.num_interactions = num_interactions
                self.cut_off = cut_off
                self.n_atoms = n_atoms
                self.out_dim = out_dim
                self.s_dim = s_dim

                self.W = nn.Linear(s_dim,s_dim, bias = True)
                self.bn_s_update = nn.BatchNorm1d(s_dim)
                self.bn_v_update = nn.BatchNorm1d(s_dim)
                self.bn_s_message = nn.BatchNorm1d(s_dim)
                self.bn_v_message = nn.BatchNorm1d(s_dim)


                self.list_message = nn.ModuleList(
                    [
                        MessageFunction(s_dim, n_atoms, out_dim, r_cut_param = cut_off)
                        for _ in range(self.num_interactions)
                    ]
                )
                self.list_update = nn.ModuleList(
                    [
                        UpdateFunction(s_dim, n_atoms, out_dim)
                        for _ in range(self.num_interactions)
                    ]
                )
                #self.list_gated_block = nn.ModuleList(
                #    [
                #        GatedBlock(s_dim, n_atoms, out_dim, r_cut_param = cut_off)
                #        for _ in range(self.num_interactions)
                #    ]
                #)

            def forward(self, s, v, r_ij, molecule_index, edge_index): # , edge_attr):

                for i in range(self.num_interactions):
                    # s = self.bn_s_message(s)
                    # v = self.bn_v_message(v)
                    delta_s, delta_v = self.list_message[i](s, v, r_ij, edge_index)
                    delta_v = delta_v # /torch.max(delta_v)
                    delta_s = delta_s # /torch.max(delta_s)
                    s, v = delta_s + s, delta_v + v
                    # print(s[0:5,1 ])
                    # print(v[0:5, 1,1])
                    # print(" shape before: ", s.shape, v.shape)

                    # s = self.bn_s_update(s)
                    # v = self.bn_v_update(v)
                    # print("shape after: ", s.shape, v.shape)

                    delta_s, delta_v = self.list_update[i](s, v)
                    # the problem is that these v's -> infinity.... so let's normalize them by their norm for example :) 
                    delta_v = (delta_v) # /torch.max(delta_v)
                    delta_s = (delta_s) # /torch.max(delta_s)
                    s, v = delta_s + s, delta_v + v
                    # print("s & v  ", s[0:2,0:2],v[0:2,0:2,0:2]) # her er det NAns..... update function that is the problem

                #list_gated_block

                s = self.W(s)
                s = F.silu(s)
                s = self.W(s)
                if torch.any(torch.isnan(s)):
                    print("s is nan --------------")
                    
                # print("s: ", s[0:5,0:2])
                sum128_s = torch.sum(s, dim = 1)
                # print("sum128_s: ", sum128_s[0:5])
                # print("molecule index: ", molecule_index, "with shape: ", len(molecule_index))
                # print(sum128_s.shape)
                # E = sum_into_tensor(sum128_s, [x + 1 for x in molecule_index]) # E is per molecule...

                # convert molecule_index to a tensor
                molecule_index = torch.tensor(molecule_index, dtype=torch.int, device=device)
                E = torch.zeros_like(torch.unique(molecule_index), device=device, dtype=torch.float32)
                E.index_add_(0, molecule_index, sum128_s)

                # print("E: ", E[0:5])
                # E = torch.Tensor(E, requires_grad=True)
                # print(E, "....E shape: ", E.shape)
                
                # print("s shape: ", s.shape) 
                # print("v shape: ", v.shape)

                # E = sum(sum(s))

                return E, s, v


        cutoff = 5 ############################## CUTOFF HERE ##########################
        # n_atoms = qm9data.dataset[0]["_n_atoms"]
        s_dim = 128
        out_dim = 128

        # model = PaiNN(s_dim, out_dim, n_atoms, cut_off=cutoff, num_interactions=3)         #Er det muligt at definere n_atoms nede i selve loopet? Nu bliver det jo det samme hver gang. Men så træner vi selvfølgelig ikke samme model?
        # optimizer = optim.Adam(model.parameters(), lr=2*1e-3, weight_decay=0)

        # def accuracy(target, pred):
        #     return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())
        
        # os.remove('split.npz')
        # qm9data2 = QM9(
        #     './qm9_kindasmall.db', 
        #     batch_size=10,
        #     num_train=50*10,
        #     num_val=20,
        #     val_batch_size=10,
        #     num_workers = 4,
        #     num_val_workers= 4,
        #     pin_memory=True,
        #     transforms=[ASENeighborList(cutoff=1.5)]
        # )
        # qm9data2.prepare_data()
        # qm9data2.setup()

        dm = QM9DataModule(target=3, subset_size= 5000, batch_size_train=10, batch_size_inference=10)
        dm.prepare_data()
        dm.setup()
        
        train_loader = dm.train_dataloader(shuffle=True)
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        print("len of train, val, test: ", len(train_loader), len(val_loader), len(test_loader))  


        from torch_geometric.datasets import QM9

        # print(next(iter(dp)))
        # train_loader = qm9data2.train_dataloader()
        
        # print("Len of train loader: ", len(train_loader))

        # # valid_set = TensorDataset(r_ij_validate, targets_validate)
        # valid_loader_2 = DataLoader(qm9data2.val_dataloader(), batch_size=10, shuffle=True, pin_memory=True)
        # valid_loader = qm9data2.val_dataloader()
        
        # print("Len of valid loader: ", len(valid_loader))

        # # test_set = TensorDataset(r_ij_test, targets_test)
        # test_loader = qm9data2.test_dataloader()

        


        def Input_function_w_device(batch, cutoff, device):
            # batchlist = batch['_idx'].tolist()
            
            r4_ij = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)  # Initialize on the specified device
            edge_index2 = []
            molecule_index = []
            counter = 0

            edge_index_batch = radius_graph(batch.pos, r=cutoff, batch = batch.batch, loop=False)

            atoms = torch.unique(batch.batch) # the unique "local" atom numbers in the batch
            atom_numbers = torch.tensor([0], device=device, dtype=torch.int32)
     
            for i in atoms:
                pos = batch.pos[batch.batch == i] # .to(device='cpu', dtype=torch.float32)
                edge_index = radius_graph(pos, r=cutoff, batch=None, loop=False)
                edge_index = edge_index.to(device=device, dtype=torch.int32)
                row, col = edge_index
                pos = pos.to(device=device, dtype=torch.float32)
                edge_distance = pos[row] - pos[col]

                r4_ij = torch.vstack((r4_ij, edge_distance))
                r4_ij = r4_ij.to(device=device, dtype=torch.float32)
                edge_index2.append(edge_index[1, :].to(device='cpu', dtype=torch.int32))
                molecule_index.extend([counter] * len(edge_distance))  # Creates a list of "internal" molecule numbers corresponding to the row in r_ij

                counter += 1
                
            r4_ij = torch.cat((r4_ij[:0], r4_ij[0 + 1:]))  # Delete the first row we used for initialization

            molecule_index_atoms = torch.tensor([0], device=device, dtype=torch.float32)


            for j in range(len(batch.y)):
                atom_numbers = torch.hstack((atom_numbers, batch.z[edge_index2[j]]))
            atom_numbers = atom_numbers[1:]  # Remove the initial zero
            n_atoms = len(atom_numbers)  # Get the total number of atoms
            # n_atoms = len(molecule_index_atoms)  # This would only be necessary if molecule_index_atoms is used
            n_atoms = n_atoms
            s0 = atom_embedding(atom_numbers)  # Ensure atom_embedding also transfers tensors to the specified device
            s0 = s0.to(device=device, dtype=torch.float32)
            v0 = torch.zeros(r4_ij.shape[0], 128, 3, device=device, dtype=torch.float32)

            

            return r4_ij, molecule_index, edge_index2, n_atoms, s0, v0, molecule_index_atoms, edge_index_batch


        ##################################### TRAINING LOOP #########################################
        num_epochs = 6
        validation_every_steps = 10
        loss_fn = nn.MSELoss() 
        cutoff = 5

        step = 0

        model = PaiNN(s_dim, out_dim, n_atoms, cut_off=cutoff, num_interactions=3)  

        device = torch.device('cpu')
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.5*1e-5, weight_decay=0.01)

        # SWA optimizer
        

        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # , could be, but we use AdamW for this... 

        # You can use custom averaging functions with `avg_fun` parameter
        swa_start = 5
        # ema_avg = lambda p_avg, p, n_avg: 0.1 * p_avg + 0.9 * p
        # ema_model = torch.optim.swa_utils.AveragedModel(model, avg_function=ema_avg)
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)

        model.train()

        train_losses = []
        valid_losses = []
        import time
        
        # print("FEEL FOR validation TARGET SIZE: ", next(iter(val_loader)).y)

                
        for epoch in range(num_epochs):
        # %matplotlib nbagg
            train_accuracies_batches = []
            
            for batch, i in zip(train_loader, range(int(len(train_loader)))):
                
                print("--- Epoch: ",epoch, "Batch: ", i, "-----")
                # batch = {k: v.to(device, dtype = torch.float32) if torch.is_tensor(v) and type(v) == float else v.to(device) for k, v in batch.items()}
                
                r4_ij, molecule_index, edge_index2, n_atoms, s0, v0, molecule_index_atoms, edge_index_batch = Input_function_w_device(batch,cutoff, device=device)
                # print(molecule_index, edge_index)
                # print("r4_ij shape: ", r4_ij.shape)
                # print("molecule_index shape: ", len(molecule_index))
                # print("edge_index2 shape: ", len(edge_index2))
                # print("n_atoms shape: ", (n_atoms))
                # print("s0 shape: ", s0.shape)
                # print("v0 shape: ", v0.shape)
                # print("molecule_index_atoms shape: ", len(molecule_index_atoms))
                # print(edge_index_batch[:,0:20])

                targets = batch.y # batch['lumo'].to(device = device, dtype=torch.float32)
                targets = torch.squeeze(targets)



                # print("len of targets: ", len(targets))
                # optimizer.zero_grad() # for every batch, since it basically is new data... NOT SURE ABOUT THIS....

                output = model(s0, v0, r4_ij, molecule_index, edge_index_batch)[0]
        
                # print("output: ", output)
                loss = loss_fn(output, targets)
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                train_losses.append(loss.item())

                if epoch > swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()


                # VALIDATION ------------------------------
                if step % validation_every_steps == 0:
                    print("VALIDATION...........")
                    
                    # Evaluate on validation set
                    # valid_losses = []
                    model.eval()
                    with torch.no_grad():
                        
                        # batch1 = next(iter(val_loader))
                        # r4_ij, molecule_index, edge_index2, n_atoms, s0, v0, molecule_index_atoms = Input_function_w_device(batch,cutoff, device = device)
                        # output = model(s0, v0, r4_ij, molecule_index)[0]
                        # # print("output: ", output)
                        # targets = batch.y # batch['lumo'].to(device = device, dtype=torch.float32)
                        # targets = torch.squeeze(targets)
                        # loss = loss_fn(output, targets)
                        # print("loss: ", loss)
                        # print("targets: ", targets)
                        # print("output: ", output)
                        # print("end of first run.... with model.eval().....")
                        
                        
                        # startValidationTime = time.time()
                        for batch, i in zip(val_loader, range(len(val_loader))):
                            print("Validation: ", i)
                            # print("Validation time of model.eval(): ", time.time()-startValidationTime)
                            r4_ij, molecule_index, edge_index2, n_atoms, s0, v0, molecule_index_atoms, edge_index_batch = Input_function_w_device(batch,cutoff, device = device)
                            # print("molecule index: ", molecule_index)
                            # print("r4_ij: ", r4_ij)
                            # print("s0 validation: ", s0[0:5,0:5])
                            # print("v0 validation: ", v0[0:5,0:5,0:5])
                            output = model(s0, v0, r4_ij, molecule_index, edge_index_batch)[0]
                            # print("output: ", output)
                            targets = batch.y # batch['lumo'].to(device = device, dtype=torch.float32)
                            targets = torch.squeeze(targets)
                            
                            loss = loss_fn(output, targets)
                            valid_losses.append(loss.item())
                        
                        model.train()
                        

                    

                    # Update bn statistics for the swa_model at the end
                    # torch.optim.swa_utils.update_bn(train_loader, swa_model)
                        # for name, param in model.named_parameters():
                        #     if param.requires_grad:
                        #         print (name, param.data)

                    # steps = (np.arange(len(train_losses), dtype=int) + 1) * validation_every_steps

                    # plt.plot(range(0,len(train_losses)), train_losses, label='train')
                    
                    # plt.clf()
                    # new figure
                    # plt.figure(figsize=(10, 5))
                    # figure.canvas.flush_events()
                    # line1.set_xdata(range(0, len(valid_losses)))
                    # line1.set_ydata(valid_losses)
                    # figure.canvas.draw()
                    # plt.plot(range(0, len(valid_losses)), valid_losses, label='validation')
                    # plt.xlabel('Validation steps')
                    # plt.ylabel('MSE loss')
                    # plt.legend()
                    # plt.title("(Train and ) Validation loss")
                    # plt.show(block=False)
                    # plt.show()
                    # display.clear_output(wait=False)
                    # display.display(plt.gcf())
                    

                    # Compute and print average losses
                    avg_train_loss = np.mean(train_losses)
                    avg_valid_loss = np.mean(valid_losses[-validation_every_steps:-1])
                    print(valid_losses)
                    print(f"Step {step:<5}   training loss: {avg_train_loss}")
                    print(f"             validation loss: {avg_valid_loss}")

                    train_losses = []  # Reset training losses for the next batch

        # Update bn statistics for the swa_model at the end
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        # Use swa_model to make predictions on test data 
        # preds = swa_model(test_input)

        print("Finished training.")
    profiler.print()
    profiler.open_in_browser()