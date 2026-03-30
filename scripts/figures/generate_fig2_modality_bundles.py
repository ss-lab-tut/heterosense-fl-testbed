import sys; sys.path.insert(0, '../..')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Generate real data for visualization ─────────────────────────────────────
from heterosense import ClientFactory
from heterosense import ConfigurationManager as CM
from heterosense import DatasetBuilder

cfg = CM.from_clients(
    ClientFactory.make(3, strategy='round_robin', seed=42),
    n_steps=500
)
data = DatasetBuilder(cfg.to_sim_config()).build()

# Pick UPRIGHT + ON_BED, LYING + ON_BED, ABSENT from client "0" (both)
bundles_both = data['0']
bundles_lidar = data['1']
bundles_pres  = data['2']

# Find representative frames
def find_frame(bundles, state, posture=None):
    for b in bundles:
        if b.semantic_state == state:
            if posture is None or b.posture_state == posture:
                return b
    return bundles[10]

b_upright  = find_frame(bundles_both, 'STATIONARY', 'UPRIGHT')
b_lying    = find_frame(bundles_both, 'STATIONARY', 'LYING')
b_absent   = find_frame(bundles_both, 'ABSENT')
b_abnormal = find_frame(bundles_both, 'ABNORMAL')

# ── Fig 2: ModalityBundle output examples ─────────────────────────────────────
fig = plt.figure(figsize=(12, 7))
fig.suptitle('Fig. 2.  Example ModalityBundle outputs from HeteroSense-FL 0 (client with both LiDAR and bed sensor)',
             fontsize=11, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

titles = ['STATIONARY\n(UPRIGHT)', 'STATIONARY\n(LYING)', 'ABSENT', 'ABNORMAL\n(phase 1)']
bundles_show = [b_upright, b_lying, b_absent, b_abnormal]
colors_map   = ['#1565C0', '#2E7D32', '#757575', '#B71C1C']

for col, (title, b, col_color) in enumerate(zip(titles, bundles_show, colors_map)):
    # Top row: point cloud (side view: x–z)
    ax_pc = fig.add_subplot(gs[0, col], projection='3d')
    if b.lidar is not None and len(b.lidar) > 0:
        pts = b.lidar
        ax_pc.scatter(pts[:,0], pts[:,1], pts[:,2],
                      c=pts[:,2], cmap='Blues', s=4, alpha=0.7)
        ax_pc.set_zlim(0, 2.5)
    ax_pc.set_title(title, fontsize=9, color=col_color, fontweight='bold')
    ax_pc.set_xlabel('x', fontsize=7); ax_pc.set_zlabel('z (m)', fontsize=7)
    ax_pc.tick_params(labelsize=6)
    if b.lidar is None:
        ax_pc.text2D(0.5, 0.5, 'None\n(modality absent)', transform=ax_pc.transAxes,
                     ha='center', va='center', fontsize=8, color='#999999')

    # Bottom row: pressure map
    ax_pr = fig.add_subplot(gs[1, col])
    if b.pressure is not None:
        im = ax_pr.imshow(b.pressure, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax_pr, fraction=0.046, pad=0.04)
    else:
        ax_pr.text(0.5, 0.5, 'None\n(no bed sensor)', transform=ax_pr.transAxes,
                   ha='center', va='center', fontsize=8, color='#999999')
        ax_pr.set_facecolor('#F5F5F5')
    ax_pr.set_title('pressure map (16×16)', fontsize=8)
    ax_pr.tick_params(labelsize=6)

# Row labels
fig.text(0.01, 0.72, 'LiDAR\npoint cloud', fontsize=9, va='center', ha='left',
         fontweight='bold', color='#1F4E79', rotation=90)
fig.text(0.01, 0.28, 'Bed pressure\nmap', fontsize=9, va='center', ha='left',
         fontweight='bold', color='#1B5E20', rotation=90)

plt.savefig('/tmp/fig2_modality_bundles.png', dpi=130, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ Fig 2 saved")