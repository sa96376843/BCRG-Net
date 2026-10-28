import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from figure_data import METHODS, SEEDS, SHORT, COL, rows, save, style
style()

groups = [SHORT[m] for m in METHODS]
data = [np.array([r['dice'] for s in SEEDS for r in rows(m,s)]) for m in METHODS]
assert all(len(d)==447 and np.isfinite(d).all() for d in data)

fig, ax = plt.subplots(figsize=(3.43, 2.85))

# ★ 淡色填充 + 原色边框小提琴
parts = ax.violinplot(data, positions=range(len(groups)), showmeans=False,
                      showmedians=False, showextrema=False)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(COL[METHODS[i]])
    pc.set_edgecolor(COL[METHODS[i]])
    pc.set_linewidth(.8)
    pc.set_alpha(.35)

# 叠加窄箱线图
bp = ax.boxplot(data, positions=range(len(groups)), widths=0.14, patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor='white', edgecolor='#2F4F3A', linewidth=.75),
                medianprops=dict(color='#2F4F3A', linewidth=1),
                whiskerprops=dict(color='#2F4F3A', linewidth=.7),
                capprops=dict(color='#2F4F3A',linewidth=.7))

# 均值菱形 + 数值标注
for i, d in enumerate(data):
    ax.scatter(i, d.mean(), marker='D', s=18, color=COL[METHODS[i]],
               edgecolor='white', linewidth=.6, zorder=5)

# The repeated case-seed values are descriptive; inferential tests belong in Table 1.

ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, fontsize=7.3, rotation=25, ha='right')
ax.set_ylabel('Per-image test Dice')
ax.set_ylim(-.04,1.04)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=.45, linestyle='--')
fig.tight_layout(pad=.35)
save(fig, 'fig05_per_image_dice')
