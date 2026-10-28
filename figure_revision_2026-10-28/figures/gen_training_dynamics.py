import numpy as np
import matplotlib.pyplot as plt
from figure_data import METHODS, SEEDS, NAME, COL, MARK, result, save, style
style()

x = np.arange(1, 13)
history = {m: np.array([[e['val_dice'] for e in result(m,s)['history']]
                         for s in SEEDS]) for m in METHODS}
assert all(v.shape == (3,12) for v in history.values())

fig, ax = plt.subplots(figsize=(3.43, 2.65))

ax.axhspan(0, 0.85, alpha=0.02, color=COL['bcskg'], zorder=0)

for i, m in enumerate(METHODS):
    v = history[m]
    y, lo, hi = v.mean(0), v.min(0), v.max(0)
    is_ours = m == 'bcskg'

    # Exactly one observed min--max band; nested bands would imply unavailable intervals.
    ax.fill_between(x, lo, hi, alpha=.12 if is_ours else .045,
                    color=COL[m], linewidth=0)

    # 主折线
    ax.plot(x, y, marker=MARK[m], color=COL[m],
            linewidth=1.9 if is_ours else 1.15, markersize=3 if is_ours else 2.2,
            markeredgecolor='white', markeredgewidth=.45, label=NAME[m], zorder=3)

    max_idx = np.argmax(y)
    if is_ours:
        ax.annotate(f'Peak {y[max_idx]:.3f}',
                    xy=(x[max_idx], y[max_idx]),
                    xytext=(x[max_idx]-2.4, min(.84,y[max_idx]+.075)),
                    fontsize=7.5, color=COL[m],
                    arrowprops=dict(arrowstyle='->', color=COL[m], lw=.65),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=COL[m], alpha=.93, linewidth=.5))

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Dice')
ax.set_ylim(0,.86)
ax.legend(frameon=False, labelspacing=.18, handlelength=1.2,
          fontsize=6.6, loc='lower right', ncol=2)
ax.set_xticks([1,3,5,7,9,11,12])
ax.grid(alpha=.45, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=.3)
save(fig, 'fig04_training_dynamics')
