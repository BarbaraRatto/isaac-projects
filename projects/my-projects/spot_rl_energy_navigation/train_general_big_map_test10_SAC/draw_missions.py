"""Offline preview of selected mission geometry; these are NOT learned trajectories."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_paths import background


def main():
    meta=json.loads((HERE/'artifacts/map_metadata.json').read_text())
    examples=json.loads((HERE/'artifacts/missions.json').read_text())['examples']
    with PdfPages(HERE/'artifacts/mission_examples.pdf') as pdf:
        for stage in range(5):
            classes=['direct','right','left']+(['energy'] if stage>=2 else [])
            fig,axes=plt.subplots(1,len(classes),figsize=(5*len(classes),5),squeeze=False)
            for ax,category in zip(axes[0],classes):
                path=np.asarray(examples[f'evaluation/{stage}/{category}'])
                a,b=path[0],path[-1]
                background(ax,meta)
                ax.plot([a[0],b[0]],[a[1],b[1]],'--',color='#777',lw=1,label='Linea diretta')
                ax.plot(path[:,0],path[:,1],color='#155da2',lw=2,label='Riferimento geometrico')
                ax.scatter(*a,c='#16a34a',s=80,marker='o',edgecolors='black',zorder=5,label='Start')
                ax.scatter(*b,c='#dc2626',s=130,marker='*',edgecolors='black',zorder=5,label='Goal')
                if stage==0:
                    d=(b-a)/np.linalg.norm(b-a);rotation=np.array([[d[0],d[1]],[-d[1],d[0]]])
                    corners=np.array([[-2.25,-2.25],[2.25,-2.25],[2.25,2.25],[-2.25,2.25]])@rotation+a
                    ax.add_patch(Polygon(corners,fill=False,edgecolor='black',linestyle=':',label='Estensione grid locale'))
                lo=path.min(0)-1.25;hi=path.max(0)+1.25
                ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_title(f'Fase {stage}: {category}')
            axes[0,0].legend(fontsize=7,loc='best')
            fig.suptitle('Test10 — esempi di valutazione. Percorsi di riferimento, non traiettorie della policy.')
            fig.tight_layout();pdf.savefig(fig,bbox_inches='tight')
            if stage==0:fig.savefig(HERE/'artifacts/mission_examples_stage0.png',dpi=150,bbox_inches='tight')
            plt.close(fig)


if __name__=='__main__':main()
