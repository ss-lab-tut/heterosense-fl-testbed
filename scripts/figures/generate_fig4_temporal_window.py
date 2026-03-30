import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Fig 4: TemporalWindowSampler (fixed layout) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
fig.suptitle(
    'Fig. 4.  TemporalWindowSampler: sliding-window interface for temporal feature extraction',
    fontsize=9.5, fontweight='bold', y=1.01)

# ── Left panel ──
ax = axes[0]
ax.set_xlim(-0.5, 11); ax.set_ylim(-1.5, 3.8); ax.axis('off')
ax.set_title('(a) Window sliding over ModalityBundle sequence', fontsize=9)

for i in range(10):
    rect = mpatches.FancyBboxPatch((i+0.05, 0.1), 0.85, 0.8,
        boxstyle="round,pad=0.05", linewidth=1,
        edgecolor='#1565C0', facecolor='#BBDEFB')
    ax.add_patch(rect)
    ax.text(i+0.48, 0.5, f'B{i}', ha='center', va='center', fontsize=8, color='#1565C0')

for i in [4, 5, 6]:
    rect = mpatches.FancyBboxPatch((i+0.05, 0.05), 0.85, 0.9,
        boxstyle="round,pad=0.05", linewidth=2.5,
        edgecolor='#E65100', facecolor='#FFE0B2')
    ax.add_patch(rect)
    ax.text(i+0.48, 0.5, f'B{i}', ha='center', va='center', fontsize=8, fontweight='bold', color='#E65100')

ax.annotate('', xy=(4.05, -0.2), xytext=(6.90, -0.2),
    arrowprops=dict(arrowstyle='<->', color='#E65100', lw=1.8))
ax.text(5.48, -0.55, 'window = 3', ha='center', fontsize=9, color='#E65100', fontweight='bold')
ax.text(5.48, 1.25, 'label from B5\n(centre bundle)', ha='center', fontsize=8, color='#E65100')

ax.annotate('', xy=(10.2, 0.5), xytext=(-0.1, 0.5),
    arrowprops=dict(arrowstyle='->', color='#888', lw=1))
ax.text(5.0, -1.2, 'time →', ha='center', fontsize=9, color='#555')

# ── Right panel ──
ax2 = axes[1]
ax2.set_xlim(0, 7); ax2.set_ylim(0, 5.0); ax2.axis('off')
ax2.set_title('(b) Feature extraction from window', fontsize=9)

rect_w = mpatches.FancyBboxPatch((0.1, 2.8), 2.0, 1.6,
    boxstyle="round,pad=0.08", linewidth=2, edgecolor='#E65100', facecolor='#FFE0B2')
ax2.add_patch(rect_w)
ax2.text(1.1, 3.6, 'Window\n[B4, B5, B6]', ha='center', va='center',
    fontsize=9, fontweight='bold', color='#E65100')

feats = [
    (4.0, '#1565C0', 'lidar_z_series(w)',   'shape (window,)'),
    (3.3, '#2E7D32', 'pressure_series(w)',  'shape (window,)'),
    (2.6, '#6A1B9A', 'center_label(w, idx)','e.g. STATIONARY'),
]
for y, col, fn, desc in feats:
    ax2.annotate('', xy=(2.4, y), xytext=(2.1, y),
        arrowprops=dict(arrowstyle='->', color='#666', lw=1.4))
    r = mpatches.FancyBboxPatch((2.4, y-0.22), 4.4, 0.5,
        boxstyle="round,pad=0.04", linewidth=1.2, edgecolor=col, facecolor=col+'22')
    ax2.add_patch(r)
    ax2.text(4.6, y+0.01, fn, ha='center', va='center', fontsize=8.2, fontweight='bold', color=col)
    ax2.text(4.6, y-0.16, f'→  {desc}', ha='center', va='center', fontsize=7.5, color=col)

rect_r = mpatches.FancyBboxPatch((0.1, 0.5), 6.7, 1.0,
    boxstyle="round,pad=0.06", linewidth=1.8, edgecolor='#1B5E20',
    facecolor='#E8F5E9', linestyle='--')
ax2.add_patch(rect_r)
ax2.text(3.45, 1.35, '↓ replace with your own encoder',
    ha='center', fontsize=8, color='#555', style='italic')
ax2.text(3.45, 0.95, 'custom_encoder(window)  →  any temporal model',
    ha='center', fontsize=8.5, color='#1B5E20', fontweight='bold')
ax2.text(3.45, 0.65, '(LSTM / Transformer / handcrafted features)',
    ha='center', fontsize=8.0, color='#1B5E20')

plt.tight_layout()
plt.savefig('/tmp/fig4_temporal_window.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Fig 4 saved")