import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ── Fig 1: Architecture diagram ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

colors = {'layer': '#1F4E79', 'module': '#2E75B6', 'api': '#1B5E20',
          'arrow': '#555555', 'bg': '#F0F4F8'}

def box(ax, x, y, w, h, label, sublabel='', color='#2E75B6', fontsize=10):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.08", linewidth=1.5,
        edgecolor=color, facecolor=color + '22')
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2 + (0.12 if sublabel else 0), label,
        ha='center', va='center', fontsize=fontsize, fontweight='bold', color=color)
    if sublabel:
        ax.text(x+w/2, y+h/2 - 0.18, sublabel,
            ha='center', va='center', fontsize=8, color='#444444', style='italic')

def layer_bg(ax, y, h, label, color):
    rect = mpatches.FancyBboxPatch((0.2, y), 9.6, h,
        boxstyle="round,pad=0.05", linewidth=2,
        edgecolor=color, facecolor=color + '11', zorder=0)
    ax.add_patch(rect)
    ax.text(0.42, y + h/2, label, ha='left', va='center',
        fontsize=9, fontweight='bold', color=color,
        rotation=90 if h > 1.2 else 0)

# Layer backgrounds
layer_bg(ax, 0.3, 1.4, 'Latent\nWorld', '#78909C')
layer_bg(ax, 1.9, 1.9, 'Observation', '#1F4E79')
layer_bg(ax, 4.0, 1.7, 'Interface', '#1B5E20')

# Latent world layer
box(ax, 1.0, 0.5, 3.0, 0.9, 'BehaviorModel', '(v1.0, unmodified)', '#78909C', 9)
box(ax, 4.5, 0.5, 4.2, 0.9, 'LatentState',
    'state · posture · bed_zone · abnormal_phase', '#78909C', 9)

# Observation layer
box(ax, 1.0, 2.1, 2.6, 0.7, 'ObservationModel', 'point cloud (N,3) · pressure (16,16)', '#1F4E79', 9)
box(ax, 3.9, 2.1, 2.0, 0.7, 'BehaviorModel', '+ support_state', '#1F4E79', 9)
box(ax, 6.2, 2.1, 2.5, 0.7, 'DatasetBuilder', '{client_id: [ModalityBundle]}', '#1F4E79', 9)

# Interface layer
box(ax, 0.7, 4.2, 2.4, 1.2, 'ClientFactory', 'N clients\nround_robin/uniform\nexplicit/random', '#1B5E20', 9)
box(ax, 3.4, 4.2, 2.5, 1.2, 'TemporalWindow\nSampler', 'window=k\nplug-in point', '#1B5E20', 9)
box(ax, 6.2, 4.2, 2.5, 1.2, 'ConfigurationManager\n.from_clients()', 'YAML-free config', '#1B5E20', 9)

# Arrows (vertical, between layers)
arrowprops = dict(arrowstyle='->', color='#555555', lw=1.5)
for x in [2.3, 5.0, 7.3]:
    ax.annotate('', xy=(x, 2.1), xytext=(x, 1.4),
        arrowprops=arrowprops)
    ax.annotate('', xy=(x, 4.2), xytext=(x, 2.8),
        arrowprops=arrowprops)

# Labels
ax.text(5, 5.9, 'HeteroSense-FL 1.0.0 — System Architecture',
    ha='center', va='center', fontsize=12, fontweight='bold', color='#1F4E79')

# Legend
for label, color in [('Latent World Layer (v1.0)', '#78909C'),
                      ('Observation Layer', '#1F4E79'),
                      ('Interface Layer', '#1B5E20')]:
    patch = mpatches.Patch(color=color + '44', label=label,
        linewidth=1.5, edgecolor=color)

patches = [mpatches.Patch(facecolor=c+'33', edgecolor=c, label=l)
           for l, c in [('Latent World Layer (v1.0 core)', '#78909C'),
                        ('Observation Layer', '#1F4E79'),
                        ('Interface Layer', '#1B5E20')]]
ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.8)

plt.tight_layout()
plt.savefig('/tmp/fig1_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ Fig 1 saved")