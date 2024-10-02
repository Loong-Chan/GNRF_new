import torch
from torch import nn
from torch_geometric.utils import add_self_loops, degree


class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_layer, dropout):
        super().__init__()
        layers = []
        in_dims = [n_in] + [n_hid] * (n_layer - 1)
        out_dims = [n_hid] * (n_layer - 1) + [n_out]
        for (in_, out_) in zip(in_dims, out_dims):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_, out_))          
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*(layers[:-1])) 
                
    def forward(self, x):
        return self.layers(x)


def accuary(predict, target):
    _, predicted_classes = torch.max(predict, 1)
    correct = (predicted_classes == target).sum().item()
    total = target.size(0)
    accuracy = correct / total
    return accuracy


def load_best_args(args):
    import json
    dataset = args.dataset
    with open("best_args.json", 'r') as f:
        json_data = json.load(f)
    if dataset in json_data:
        for key, value in json_data[dataset].items():
            if not hasattr(args, key):
                setattr(args, key, value)
    return args


def dirichlet_energy(x, edge_index):
    edge_index = add_self_loops(edge_index)[0]
    deg = degree(edge_index[0]).reshape(-1, 1)
    x = x * deg**(-0.5)
    x_i = x[edge_index[0]]
    x_j = x[edge_index[1]]
    energy = 0.5 * ((x_i - x_j)**2).sum()
    return energy
