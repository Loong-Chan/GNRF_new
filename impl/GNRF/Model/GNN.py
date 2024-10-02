import torch
from torch import nn
import torch.nn.functional as F

from Utils import MLP


class GNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bn_in = nn.BatchNorm1d(args.num_hid)
        self.bn_out = nn.BatchNorm1d(args.num_hid)
        self.mlp_in = MLP(args.num_hid, args.num_hid, args.num_hid, 2, args.dropout)
        self.mlp_out = MLP(args.num_hid, args.num_hid, args.num_hid, 2, args.dropout)
        self.lin_in = nn.Linear(args.num_feat, args.num_hid)
        self.lin_out = nn.Linear(args.num_hid, args.num_class)
        self.ODE_block = self.get_ODE_block(args)
        self.t = torch.tensor([args.t_start, args.t_end]).to(args.device)

        if args.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint as odeint
        self.odeint = odeint

    def get_ODE_block(self, args):
        ode = args.ode
        if ode == "GNRF":
            from Model.GNRF import GNRF as ode_block
        else:
            raise NotImplementedError
        return ode_block(args)

    def pre_transform(self, x, edge_index):
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.lin_in(x)
        x = F.relu(x)
        if self.args.use_mlp_in:
            x = self.mlp_in(x)
            x = F.relu(x)
        if self.args.use_bn_in:
            x = self.bn_in(x)
        return x

    def solve_ODE(self, x_0, edge_index, t=None):
        T = self.t if t is None else t
        self.ODE_block.set_edges(edge_index)
        output = self.odeint(
            func = self.ODE_block,
            y0 = x_0,
            t = T,
            atol = self.args.tol_scale * 1e-7,
            rtol = self.args.tol_scale * 1e-9,
            method=self.args.solver
        )
        return output

    def post_transform(self, x, edge_index):
        x = F.relu(x)
        if self.args.use_mlp_out:
            x = self.mlp_out(x)
            x = F.relu(x)
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.lin_out(x)
        x = F.log_softmax(x, dim=1)
        return x

    def forward(self, x, edge_index):
        x = self.pre_transform(x, edge_index)
        x = self.solve_ODE(x, edge_index)[-1]
        x = self.post_transform(x, edge_index)
        return x
    