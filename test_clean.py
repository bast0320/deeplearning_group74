import torch
import torch.nn as nn
import torch.nn.functional as Func

from PaiNN_clean import PaiNN
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
import numpy as np
from typing import Optional, Union, List
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import radius, radius_graph
import matplotlib.pyplot as plt
from torch import optim
import pickle
import datetime
# Import data
class GetTarget(BaseTransform):
    def __init__(self, target: Optional[int] = None, mean: float = 0.0, std: float = 1.0) -> None:
        self.target = [target]
        self.mean = mean
        self.std = std

    def forward(self, data: Data) -> Data:
        if self.target is not None:
            # data.y = data.y[:, self.target]
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

class AtomEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(AtomEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim) # main line

    def forward(self, atomic_numbers):
        return self.embedding(atomic_numbers)

# Example usage: here we initialize an instance of the class, which can be used to embed atom numbers as a function

if __name__ == "__main__":

    
    atom_embedding = AtomEmbedding(num_embeddings=30, embedding_dim=128)

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
    batch_size_input = 100
    dm = QM9DataModule(target=3, batch_size_train=batch_size_input, batch_size_inference=batch_size_input, seed = 1)
    dm.prepare_data()
    dm.setup()
    cutoff = 5
    print("TARGET: ",dm.target, "which is: ", dictionary_targets[dm.target] )

    train_loader = dm.train_dataloader(shuffle=False)
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()


    print("Len of train, val and test loaders: ", len(train_loader), len(val_loader), len(test_loader))

    # Paramerts
    # F: Num. features, r_ij: cartesian positions
    F = 128

    batch_sample = next(iter(train_loader))
    
    r_ij = batch_sample.pos # positions
    # print("r ij shape: ", r_ij.shape)
    num_nodes = r_ij.shape[0]

    s0 = atom_embedding(batch_sample.z)  # Ensure atom_embedding also transfers tensors to the specified device
    v0 = torch.zeros(r_ij.shape[0], 128, 3, dtype=torch.float32)

    # edge_attr: inter_atomic distances
    edge_index = radius_graph(r_ij, r= cutoff , batch=batch_sample.batch, loop=False)
    row, col = edge_index
    edge_attr = r_ij[row] - r_ij[col]

    num_epochs = 10
    test = True # Test = True is for loading a saved model and evaluating it on the test set. 
    # test = False is for training a new model.

    from pyinstrument import Profiler
    with Profiler(interval=0.1) as profiler:
        loss_fn = nn.MSELoss() 

        if test == "Temp":
            targets_all = []
            print((train_loader.dataset[86000:86100].y))
            for batch, i in zip(train_loader, range(int(len(train_loader)))):
                targets_all.append(batch.y)
                if any(batch.y > 3.5):
                    print("Batch: ", i, "contains a target > 3.5")
                    print("Target: ", batch.y)

            targets_all = torch.cat(targets_all, dim=0)
            plt.figure(figsize=(10,5))
            plt.plot(targets_all)
            # plt.show()


        if test == False:
            model = PaiNN(F, F, 4)
            # model.load_state_dict(torch.load("model_fulldata.pt"))
           
            form = model(dm.target, s0, v0, edge_index, edge_attr)
            # print("Model parameters: ", model.parameters())
            optimizer = optim.AdamW(model.parameters(), lr=5*1e-4, weight_decay=0.05)
            # print(form[0].shape)
            train_losses = []
            val_losses = []

            for epoch in range(num_epochs):
                step = 0
                
                validation_every_steps = 200
                
                
                # torch.save(model.state_dict(), "model_full_epochs.pt")


                for batch, i in zip(train_loader, range(int(len(train_loader)))):
                    print("Epoch: ", epoch, "Batch: ", i)
                    if any(batch.y > 3.5):
                        print("Batch: ", i, "contains a target > 3.5")
                        print("Target: ", batch.y)
                        continue
                    r_ij = batch.pos # positions
                   
                    num_nodes = r_ij.shape[0]

                    s0 = atom_embedding(batch.z)  
                    v0 = torch.zeros(r_ij.shape[0], 128, 3, dtype=torch.float32)
                    edge_index = radius_graph(r_ij, r= cutoff , batch=batch.batch, batch_size= batch_size_input, loop=False)

                    row, col = edge_index
                    edge_attr = r_ij[row] - r_ij[col]
                    output_lengthOfr_ij = torch.sum(model(dm.target, s0, v0, edge_index, edge_attr), dim = 1)

                    
                    targets = torch.squeeze(batch.y)
                    output = torch.zeros_like(targets, dtype=torch.float32)
                    output.index_add_(0, batch.batch, output_lengthOfr_ij)

                    # converting to correct units, eV -> meV if applicable
                    output = dm.unit_conversion[dm.target](output)
                    targets = dm.unit_conversion[dm.target](targets)
                    
                    loss = loss_fn(output, targets)
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_losses.append(loss.item())

                    step += 1

                    if step % validation_every_steps == 0:
                        val_loss = 0
                        model.eval()
                        print("Validation at step: ", step)
                        with torch.no_grad():
                            for batch in val_loader:
                              
                                r_ij = batch.pos # positions
                            
                                num_nodes = r_ij.shape[0]

                                s0 = atom_embedding(batch.z)  
                                v0 = torch.zeros(r_ij.shape[0], 128, 3, dtype=torch.float32)
                                edge_index = radius_graph(r_ij, r= cutoff , batch=batch.batch, batch_size= batch_size_input, loop=False)

                                row, col = edge_index
                                edge_attr = r_ij[row] - r_ij[col]
                                output_lengthOfr_ij = torch.sum(model(dm.target, s0, v0, edge_index, edge_attr), dim = 1)
                                targets = torch.squeeze(batch.y)
                                output = torch.zeros_like(targets, dtype=torch.float32)
                                output.index_add_(0, batch.batch, output_lengthOfr_ij)

                                # converting to correct units, eV -> meV if applicable
                                output = dm.unit_conversion[dm.target](output)
                                targets = dm.unit_conversion[dm.target](targets)

                                loss = loss_fn(output, targets)
                                
                                # val_loss += loss.item()
                                val_losses.append(loss.item())
                        model.train()
                        
                        print("Step: ", step, "Validation loss mean of last 50: ", np.mean(val_losses[-50:-1]))
            # save val_losses and train_losses as pickle
            # create a string with the name of the file + current datetime
            name_val = "val_losses_big" + str(datetime.datetime.now()) + ".pkl"
            name_train = "train_losses_big" + str(datetime.datetime.now()) + ".pkl"
            with open(name_val, "wb") as f:
                pickle.dump(val_losses, f)
            with open(name_train, "wb") as f:
                pickle.dump(train_losses, f)


            print("Finished training!")

            # save the model
            model_name = "model_fulldata" + str(datetime.datetime.now()) + ".pt"
            torch.save(model.state_dict(), model_name)

            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.plot(train_losses, label="Training loss")
            plt.subplot(1,2,2)
            plt.plot(val_losses, label="Validation loss")
            plt.show()

            
        if test == True:
            loss_fn = nn.L1Loss()
            
            test_losses = []
            test_loss = 0
            model = PaiNN(F, F, 4)
            # model.load_state_dict(torch.load("model_fulldata.pt"))
            model.load_state_dict(torch.load("model_fulldata2023-12-03 18:00:14.019695.pt"))
            
            model.eval()
            with torch.no_grad():
                for batch, i in zip(test_loader, range(len(test_loader))):
                    print("Batch: ", i)
                    r_ij = batch.pos # positions
                            
                    num_nodes = r_ij.shape[0]

                    s0 = atom_embedding(batch.z)  
                    v0 = torch.zeros(r_ij.shape[0], 128, 3, dtype=torch.float32)
                    edge_index = radius_graph(r_ij, r= cutoff , batch=batch.batch, batch_size= batch_size_input, loop=False)

                    row, col = edge_index
                    edge_attr = r_ij[row] - r_ij[col]
                    output_lengthOfr_ij = torch.sum(model(dm.target, s0, v0, edge_index, edge_attr), dim = 1)
                    targets = torch.squeeze(batch.y)
                    output = torch.zeros_like(targets, dtype=torch.float32)
                    output.index_add_(0, batch.batch, output_lengthOfr_ij)

                    # converting to correct units, eV -> meV if applicable
                    output = dm.unit_conversion[dm.target](output)
                    targets = dm.unit_conversion[dm.target](targets)

                    loss = loss_fn(output, targets)
                    if i == 1:
                        print("Output: ", output, "Target: ", targets, "Error: ", loss.item())
                        
                    # test_loss += loss.item()
                    test_losses.append(loss.item())
            
            name_test = "test_losses_big" + str(datetime.datetime.now()) + ".pkl"
            with open(name_test, "wb") as f:
                pickle.dump(test_losses, f)
            print("Test loss MAE average: ", np.mean(test_losses))
            plt.figure(figsize=(10,5))
            plt.plot(test_losses, label="Test loss in MAE")
            plt.title("Test loss (MAE) for PaiNN. Target: " + dictionary_targets[dm.target])
            plt.show()
    profiler.print()
    # profiler.open_in_browser()
            