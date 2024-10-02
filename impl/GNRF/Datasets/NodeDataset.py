import torch
from torch_geometric.datasets import Planetoid, WebKB, HeterophilousGraphDataset, CitationFull
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import remove_self_loops, to_undirected
from pytorch_lightning import seed_everything


from Datasets.Constants import *




class NodeDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.file_root = DATASET_ROOT
        if self.dataset in ['pubmed']:
            data = Planetoid(root=self.file_root, name=self.dataset )
        elif self.dataset in ['cornell', 'texas', 'wisconsin']:
            data = WebKB(root=self.file_root, name=self.dataset )
        elif self.dataset in ["Roman-empire", "Tolokers"]:
            data = HeterophilousGraphDataset(root=self.file_root, name=self.dataset)
        elif self.dataset in ["Cora_Full"]:
            data = CitationFull(root=self.file_root, name="Cora")
        elif self.dataset in ["ogbn-arxiv"]:
            data = PygNodePropPredDataset("ogbn-arxiv", root=self.file_root+"/ogbn_arxiv/")
            data.y = data.y.flatten()

        self.x = data.x
        self.edge_index = data.edge_index
        self.y = data.y

        self.nfeat = self.x.shape[1]
        self.nclass = self.y.max().item() + 1
        self.nnode = self.x.shape[0]
    
        self.edge_index = remove_self_loops(self.edge_index)[0]
        self.edge_index = to_undirected(self.edge_index)

        self.x[self.x.isnan()] = 0.
        rowsum = self.x.sum(dim=1, keepdim=True)
        rowsum[rowsum == 0.] = 1.
        self.x = self.x / rowsum

    def random_split(self, seed, p_train=0.6, p_val=0.2):
        seed_everything(seed)
        n_train = int(self.nnode * p_train)
        n_val = int(self.nnode * p_val)
        full_idx = torch.randperm(self.nnode)
        train_idx = full_idx[:n_train]
        val_idx = full_idx[n_train+1:n_train+n_val]
        test_idx = full_idx[n_train+n_val+1:]
        return train_idx, val_idx, test_idx
