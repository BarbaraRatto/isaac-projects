"""Export the prepared terrain layout, with one-based cell coordinates; no simulation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

HERE = Path(__file__).resolve().parent
COLORS = {'t1_asphalt':('#d6dce1','Asfalto'), 't2_slippery':('#b6e3f2','Ghiaccio'),
          't3_ramp':('#ede0aa','Rampe'), 't4_fine_gravel':('#ead0b3','Ghiaia'),
          't5_large_rocks':('#bfc6af','Rocce'), 't6_obstacles':('#d8ad95','Ostacoli'),
          't7_stairs':('#d4c1e6','Scale')}


def main():
    meta = json.loads((HERE/'artifacts/map_metadata.json').read_text())
    fig, ax = plt.subplots(figsize=(14, 8))
    w, h = meta['cell_width_m'], meta['cell_height_m']
    x0, y0 = meta['bounds_min'][:2]
    for r, row in enumerate(meta['grid']):
        for c, name in enumerate(row):
            color, label = COLORS[name]
            ax.add_patch(Rectangle((x0+c*w, y0+r*h), w, h, facecolor=color, edgecolor='#555', lw=.8))
            ax.text(x0+(c+.5)*w, y0+(r+.5)*h, f'{label}\n({r+1},{c+1})',
                    ha='center', va='center', fontsize=10, weight='bold' if name=='t2_slippery' else 'normal')
    ax.set(xlim=(x0, x0+7*w), ylim=(y0, y0+7*h), xlabel='X [m] → colonne', ylabel='Y [m] → righe')
    ax.set_aspect('equal')
    ax.set_title('Test 9 SAC — mappa di training\nRiga 1 in basso; ghiaccio separato anche in diagonale', pad=16)
    fig.legend(handles=[Patch(facecolor=c, label=n) for c,n in COLORS.values()],
               loc='lower center', bbox_to_anchor=(.5,.055), ncol=7)
    fig.text(.5, .025, 'Spawn e goal: esclusi ghiaccio e scale. Ogni cella misura 6 × 3 m.', ha='center')
    fig.tight_layout(rect=(0,.12,1,1))
    for suffix in ('png','pdf'):
        fig.savefig(HERE/f'artifacts/map_layout.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
