from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    base = Path(__file__).resolve().parent
    obj = base / 'optimized_points.obj'
    out = base / 'point_cloud_preview.png'

    verts = []
    colors = []
    with obj.open('r', encoding='utf-8') as file:
        for line in file:
            if not line.startswith('v '):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([float(parts[4]), float(parts[5]), float(parts[6])])

    verts = np.asarray(verts, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if verts.size == 0:
        raise RuntimeError('No vertices found in optimized_points.obj')

    n = min(len(verts), 8000)
    indices = np.random.default_rng(0).choice(len(verts), size=n, replace=False)
    points = verts[indices]
    cols = colors[indices]

    fig = plt.figure(figsize=(7, 7), dpi=180)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=cols, s=0.8, linewidths=0)
    ax.set_title('Optimized Point Cloud Preview')
    ax.set_axis_off()
    ax.view_init(elev=15, azim=135)
    plt.tight_layout(pad=0)
    plt.savefig(out, bbox_inches='tight', pad_inches=0.02)
    print(out)


if __name__ == '__main__':
    main()
