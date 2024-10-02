import argparse

parser = argparse.ArgumentParser()

# Command Parameters
parser.add_argument("--dataset", default="pubmed")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--rewiring", default=None, choices=[None, "fosr", "sdrf"])

# Training Setting
parser.add_argument("--trial", default=3, type=int)
parser.add_argument("--verbose", default=True, type=bool)
parser.add_argument("--epoch", default=500, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--dropout", default=0.5, type=float)

# Network Setting
parser.add_argument("--edgenet", default=True, type=bool)
parser.add_argument("--damping", default=True, type=bool)
parser.add_argument("--num_hid", default=64, type=int)
parser.add_argument("--use_mlp_in", default=False, type=bool)
parser.add_argument("--use_mlp_out", default=False, type=bool)
parser.add_argument("--use_bn_in", default=True, type=bool)
parser.add_argument("--use_bn_out", default=True, type=bool)

# ODE Setting
parser.add_argument("--ode", default="GNRF")
parser.add_argument("--t_start", default=1e-5, type=float)
parser.add_argument("--t_end", default=1, type=float)
parser.add_argument("--tol_scale", default=1, type=float)
parser.add_argument("--solver", default="implicit_adams")
parser.add_argument("--adjoint", default=False, type=bool)
parser.add_argument("--channel_curv", default=True, type=bool)
args = parser.parse_args()
