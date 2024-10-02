import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from Utils import MLP


class GNRF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.edge_index = None
        self.damping = args.damping
        self.edgenet = args.edgenet
        if self.edgenet:
            self.mlp_1 = MLP(2 * args.num_hid, args.num_hid, args.num_hid, 2, args.dropout)
            if args.channel_curv: 
                self.mlp_2 = MLP(2 * args.num_hid, args.num_hid, args.num_hid, 2, args.dropout)
            else:
                self.mlp_2 = MLP(2 * args.num_hid, args.num_hid, 1, 2, args.dropout)
        else:
            self.a = nn.Parameter(torch.tensor(0.5))

    def set_edges(self, edge_index):
        self.edge_index = edge_index

    def forward(self, t, H):
        if self.damping:
            norm = torch.norm(H, p=2, dim=1, keepdim=True) + 1e-8
            H = H / norm
        H_i = H[self.edge_index[0]]
        H_j = H[self.edge_index[1]]
        if self.edgenet:
            curv = self.curvature(H_i, H_j)
        else:
            curv = torch.clamp(self.a, 1e-8, 1)
        if self.damping:
            cos = (H_i * H_j).sum(dim=1, keepdim=True)
            H_edge = curv * (H_j - cos * H_i)
        else:
            H_edge = curv * (H_j - H_i)
        H = scatter(H_edge, self.edge_index[0], reduce="mean")
        if self.damping:
            norm = torch.norm(H, p=2, dim=1, keepdim=True) + 1e-8
            H =  H / norm
        return H

    def curvature(self, H_i, H_j):
        curv = torch.cat((H_i, H_j), dim=1)
        curv = F.relu(self.mlp_1(curv))
        curv = scatter(curv, self.edge_index[0])
        curv = torch.cat((curv[self.edge_index[0]], curv[self.edge_index[1]]), dim=1)
        curv = self.mlp_2(curv)
        return curv
