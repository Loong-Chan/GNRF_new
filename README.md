# Graph Neural Ricci Flow: Evolving Feature from a Curvature Perspective

## Requirement:

+  Python 3.11.5
+ PyTorch 2.1.1
+ PyTorch Geometric 2.4.0 
+ Torchdiffeq 0.2.4

## Run codes:

```
cd impl
cd Model

# demo for running main experiment 
python Main.py --dataset Roman-empire  --device cpu --verbose True 

# demo for running ablation study
python Main.py --dataset Roman-empire  --device cpu --verbose True --edgenet False

```

