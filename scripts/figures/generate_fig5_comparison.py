import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Fig 5: Comparison table (wider, no text overflow) ────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.axis('off')
fig.suptitle(
    'Fig. 5.  HeteroSense-FL in context: feature comparison with existing FL tools simulation tools',
    fontsize=9.5, fontweight='bold', y=1.0)

cols = ['Flower [3]', 'FedML [4]', 'PTB-FLA [5]', 'HeteroSense-FL\n(this workrk)']
rows = [
    'Structured sensor observations\n(point cloud / pressure map)',
    'Per-client modality\nheterogeneity (N clients)',
    'LiDAR point cloud\ngeneration',
    'Bed pressure map\ngeneration',
    'Temporal window\nsampler interface',
    'Reproducible benchmark\nscripts included',
]

data = [
    ['✗', '✗', '✗', '✓'],
    ['✗', '✗', '✗', '✓'],
    ['✗', '✗', '✗', '✓'],
    ['✗', '✗', '✗', '✓'],
    ['✗', '✗', '✗', '✓'],
    ['partial', 'partial', '✗', '✓'],
]

table = ax.table(
    cellText=data,
    rowLabels=rows,
    colLabels=cols,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(9)

# Header
for j in range(len(cols)):
    c = table[0, j]
    c.set_facecolor('#1F4E79')
    c.set_text_props(color='white', fontweight='bold', fontsize=9)

# Row labels
for i in range(len(rows)):
    c = table[i+1, -1]
    c.set_facecolor('#EEF2F6')
    c.set_text_props(fontsize=8.0)

# Data cells
for i in range(len(rows)):
    for j in range(len(cols)):
        c = table[i+1, j]
        v = data[i][j]
        if v == '✓':
            c.set_facecolor('#E8F5E9')
            c.set_text_props(color='#2E7D32', fontweight='bold', fontsize=12)
        elif v == '✗':
            c.set_facecolor('#FFF5F5')
            c.set_text_props(color='#C62828', fontsize=12)
        else:
            c.set_facecolor('#FFF8E1')
            c.set_text_props(color='#E65100', fontsize=8.5)

# Highlight HeteroSense-FL column
for i in range(len(rows)+1):
    c = table[i, len(cols)-1]
    c.set_linewidth(2.5)

plt.tight_layout()
plt.savefig('/tmp/fig5_comparison.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Fig 5 saved")