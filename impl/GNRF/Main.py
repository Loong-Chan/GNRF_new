import torch
from torch.optim import Adam
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch_geometric.utils import to_undirected, remove_self_loops
import gc

from Datasets import NodeDataset
from Utils import accuary, load_best_args
from Model.GNN import GNN



def main(args):
    data = NodeDataset(args.dataset)
    if (args.dataset in ["cornell", "wisconsin", "texas"]) and args.rewiring is not None:
        import numpy as np
        rewired = np.load("Datasets/rewiring.npz")
        edge_index = rewired[f"{args.rewiring}_{args.dataset}"]
        edge_index = torch.from_numpy(edge_index)
        edge_index = to_undirected(edge_index)
        edge_index = remove_self_loops(edge_index)[0]
        edge_index = edge_index.to(args.device)
    else:
        edge_index = data.edge_index.to(args.device)
    x = data.x.to(args.device)
    y = data.y.to(args.device)
    args.num_feat = data.nfeat
    args.num_class = data.nclass

    results = []
    for trial in range(args.trial):
        seed_everything(trial)
        train_idx, val_idx, test_idx = data.random_split(seed=trial, p_train=0.6, p_val=0.2)
        train_idx = train_idx.to(args.device)
        val_idx = val_idx.to(args.device)
        test_idx = test_idx.to(args.device)

        model = GNN(args).to(args.device)
        optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        val_loss, test_acc = [], []
        for epoch in range(args.epoch):
            ## Train
            model.train()
            output = model(x, edge_index)
            train_loss = F.nll_loss(output[train_idx], y[train_idx]) 
            optim.zero_grad()
            train_loss.backward()
            optim.step()

            ## Validation & Test
            model.eval()
            with torch.no_grad():
                output = model(x, edge_index).detach()
                val_loss.append(F.nll_loss(output[val_idx], y[val_idx]).item())
                test_acc.append(accuary(output[test_idx], y[test_idx]))
                if args.verbose:
                    print(f"[Epoch {epoch:3d}] Train Loss: {train_loss:.4f}", \
                        f"Valid Loss: {val_loss[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}.")
            gc.collect()

        best = torch.tensor(val_loss).argmin()
        best = test_acc[best]
        if args.verbose:
            print(f"Test Acc: {best:.4f}")
        results.append(best)
    results = torch.tensor(results)
    mean, std = results.mean(), results.std()
    print(f"Mean: {mean:.4f}, Std: {std:.4f}", flush=True)
    return mean


if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()
    args, unknown_args = parser.parse_known_args()
    unknown_args_dict = {}
    for i in range(0, len(unknown_args), 2):
        key = unknown_args[i].lstrip('-')
        value = unknown_args[i+1] if i+1 < len(unknown_args) else None
        unknown_args_dict[key] = value
    for key, value in unknown_args_dict.items():
        setattr(args, key, value)
    args = load_best_args(args)
    main(args)


