"""Standalone checkpoint inference. Never opens the Y field, even if present."""
import argparse
from pathlib import Path

import numpy as np
import torch

from model import STGNN_TCN


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=Path('runs/main/best.pt'))
    p.add_argument('--input', type=Path, required=True, help='NPZ with X, X_mask, node_team')
    p.add_argument('--output', type=Path, default=Path('predictions.npy'))
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg, stats = ckpt['config'], ckpt['normalization']
    torch.set_num_threads(cfg['cpu_threads'])
    net = STGNN_TCN(stats['mean'], stats['std'], **{k: cfg[k] for k in ['width', 'dropout', 'distance_scale_m']}).to(device)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    with np.load(args.input, allow_pickle=False) as z:
        x, mask, team = z['X'], z['X_mask'], z['node_team']
    assert x.shape[1:] == (50, 23, 4) and mask.shape == x.shape
    assert mask.dtype == np.bool_ and team.shape == (len(x), 23)
    assert mask[:, -1, :22].all(), 'This model requires observed player anchor position/velocity'
    assert np.isfinite(x[mask]).all() and np.isin(team, [0, 1, 2]).all()
    results = []
    with torch.inference_mode():
        for i in range(0, len(x), args.batch_size):
            xb = torch.tensor(x[i:i+args.batch_size], dtype=torch.float32, device=device)
            mb = torch.tensor(mask[i:i+args.batch_size], dtype=torch.bool, device=device)
            tb = torch.tensor(team[i:i+args.batch_size], dtype=torch.long, device=device)
            results.append(net(xb, mb, tb).cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.concatenate(results))
    print(f'Saved {len(x)} windows of +1s player XY in metres to {args.output}; device={device}')


if __name__ == '__main__':
    main()
