"""Bundle Adjustment (Task 1) - minimal PyTorch implementation.

Usage:
  python bundle_adjustment.py --n_iters 200 --lr 1e-2

Notes:
  - This is a simple, vectorized implementation that optimizes:
    * shared focal length `f`
    * per-camera Euler angles (3) and translations (3)
    * all 3D points (Nx3)
  - Observations are loaded from `data/points2d.npz` with keys `view_000`..`view_049`.
  - Colored points are saved to `optimized_points.obj`.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def euler_angles_to_matrix(e):
    # e: (..., 3) angles in radians, convention: rotate X then Y then Z
    x, y, z = e[..., 0], e[..., 1], e[..., 2]
    cx = torch.cos(x); sx = torch.sin(x)
    cy = torch.cos(y); sy = torch.sin(y)
    cz = torch.cos(z); sz = torch.sin(z)

    Rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
        torch.stack([torch.zeros_like(cx), sx, cx], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
        torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
    ], dim=-2)

    # apply X then Y then Z: R = Rz @ Ry @ Rx
    R = torch.matmul(Rz, torch.matmul(Ry, Rx))
    return R


def project_points(points, R, T, f, cx, cy):
    # points: (N,3), R: (3,3) or (K,3,3), T: (3,) or (K,3)
    # return projected (N,2) for single camera or (K,N,2) for multiple
    if R.dim() == 2:
        # single camera
        Xc = (points @ R.T) + T
        u = -f * (Xc[:, 0] / Xc[:, 2]) + cx
        v = f * (Xc[:, 1] / Xc[:, 2]) + cy
        return torch.stack([u, v], dim=-1)
    else:
        # multiple cameras: R (K,3,3), T (K,3)
        K = R.shape[0]
        N = points.shape[0]
        # (K, N, 3) = (K,3,3) @ (3,N)
        P = points.t().unsqueeze(0).expand(K, 3, N)
        Xc = torch.matmul(R, P).permute(0, 2, 1) + T.unsqueeze(1)
        u = -f.view(-1, 1) * (Xc[:, :, 0] / Xc[:, :, 2]) + cx
        v = f.view(-1, 1) * (Xc[:, :, 1] / Xc[:, :, 2]) + cy
        return torch.stack([u, v], dim=-1)  # (K, N, 2)


def save_obj(path: Path, verts: np.ndarray, colors: np.ndarray):
    with open(path, 'w', encoding='utf-8') as f:
        for (x, y, z), (r, g, b) in zip(verts, colors):
            f.write(f"v {x} {y} {z} {r} {g} {b}\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--n_iters', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--d', type=float, default=2.5, help='initial camera -Z distance')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    pts2d_np = np.load(data_dir / 'points2d.npz')
    views = sorted([k for k in pts2d_np.files if k.startswith('view_')])
    Kcam = len(views)
    sample = pts2d_np[views[0]]
    N = sample.shape[0]

    # stack observations: (K, N, 3)
    obs = np.stack([pts2d_np[v] for v in views], axis=0)
    obs_xy = torch.from_numpy(obs[:, :, :2]).float().to(device)
    vis = torch.from_numpy(obs[:, :, 2:3]).float().to(device)

    colors = np.load(data_dir / 'points3d_colors.npy')  # (N,3) in 0-255 or 0-1
    if colors.max() > 1.1:
        colors = colors / 255.0

    H = W = 1024
    cx = W / 2.0
    cy = H / 2.0

    # Initialize parameters
    torch.manual_seed(0)
    points3d = torch.randn(N, 3, device=device, dtype=torch.float32) * 0.01

    # per-camera euler and translation
    eulers = torch.zeros(Kcam, 3, device=device, dtype=torch.float32)
    translations = torch.zeros(Kcam, 3, device=device, dtype=torch.float32)
    translations[:, 2] = -args.d

    # focal length (positive)
    f = torch.tensor([800.0], device=device, dtype=torch.float32)

    # make them optimizable
    points3d = torch.nn.Parameter(points3d)
    eulers = torch.nn.Parameter(eulers)
    translations = torch.nn.Parameter(translations)
    f = torch.nn.Parameter(f)

    optimizer = torch.optim.Adam([points3d, eulers, translations, f], lr=args.lr)

    losses = []
    for it in range(args.n_iters):
        optimizer.zero_grad()
        R = euler_angles_to_matrix(eulers)  # (K,3,3)
        proj = project_points(points3d, R, translations, f, cx, cy)  # (K,N,2)

        diff = (proj - obs_xy) * vis
        loss = (diff ** 2).sum() / (vis.sum() + 1e-8)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if it % 10 == 0 or it == args.n_iters - 1:
            print(f"iter {it:4d} loss={loss.item():.6f} f={f.item():.3f}")

    # save loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=150)

    verts = points3d.detach().cpu().numpy()
    save_obj(Path('optimized_points.obj'), verts, colors)
    print('Saved optimized_points.obj and loss_curve.png')


if __name__ == '__main__':
    main()
