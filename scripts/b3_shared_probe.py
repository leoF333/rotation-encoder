"""B3 — sonde exploratoire : décodeur partagé entre images, image t = g(z·t) (ae.SharedFrameAE), template arc, seed 0.

Variantes : entrée brute (lr 3e-4 ; lr 1e-3 donne MSE distrib. 6.1e-3, gap 7.6e-3), entrée de Fourier [cos(k u), sin(k u)], K = 8, 16.
Résultats (2026-09-18) : plain ind 4.0e-3 / gap 4.4e-3 ; four8 4.5e-5 / 1.1e-2 ; four16 1.3e-3 / 1.3e-2.
Usage : python scripts/b3_shared_probe.py (from the repository root)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3_architectures import datasets, D, T, N
from aerot import ae
from aerot.render import get_device
dev = get_device()
data = datasets('arc', dev)
class Four(nn.Module):
    def __init__(s, K): super().__init__(); s.register_buffer('k', torch.arange(1, K+1, dtype=torch.float32))
    def forward(s, u): return torch.cat([torch.cos(u*s.k), torch.sin(u*s.k)], -1)
def make(variant):
    m = ae.SharedFrameAE(N, T, D)
    if variant.startswith('four'):
        K = int(variant[4:])
        m.decoder = nn.Sequential(Four(K), ae.mlp([2*K, 64, 256, 1024, D*D], 'relu', last_act='sigmoid'))
    return m
for variant, lr in [('plain', 3e-4), ('four8', 1e-3), ('four16', 1e-3)]:
    torch.manual_seed(0); m = make(variant)
    t0 = time.time(); info = ae.train(m, data['tr'][1], epochs=200, batch=128, lr=lr, device=dev, seed=0, log_every=50)
    out = {}
    for part in ['ind', 'gap']:
        z, e = ae.encode_eval(m, data[part][1], device=dev); out[part] = e.mean()
    print(variant, lr, f"ind {out['ind']:.2e} gap {out['gap']:.2e} z[{z.min():.2f},{z.max():.2f}] {time.time()-t0:.0f}s", flush=True)
