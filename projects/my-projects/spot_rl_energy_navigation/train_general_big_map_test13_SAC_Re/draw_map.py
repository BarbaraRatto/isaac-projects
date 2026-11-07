"""Export the frozen lab layout without starting Isaac Sim."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

HERE=Path(__file__).resolve().parent


def main():
    meta=json.loads((HERE/'artifacts/map_metadata.json').read_text())
    materials=json.loads((HERE/'materials.json').read_text())
    fig,ax=plt.subplots(figsize=(8,8))
    for r,row in enumerate(meta['grid']):
        for c,name in enumerate(row):
            key='black' if name=='lab_blackice' else 'grey'
            ax.add_patch(Rectangle((3*c,3*r),3,3,facecolor=materials[key]['color_srgb'],
                                   edgecolor='#666666',lw=.8))
            ax.text(3*c+1.5,3*r+1.5,f'{"Nero" if key=="black" else "Grigio"}\nr{r+1}, c{c+1}',
                    ha='center',va='center',fontsize=10,color='white' if key=='black' else 'black')
    ax.set(xlim=(0,15),ylim=(0,15),aspect='equal',xlabel='X [m]',ylabel='Y [m]',
           xticks=range(0,16,3),yticks=range(0,16,3),
           title='Test13 SAC Re · mappa fissa 15×15 m · layout richiesto\nCelle 3×3 m: 17 grigie + 8 nere')
    fig.legend(handles=[Patch(facecolor=materials[k]['color_srgb'],edgecolor='#666666',label=label)
                        for k,label in [('grey','Grigio: nessuna resistenza'),('black','Nero: ghiaccio (µs 0,10; µd 0,08)')]],
               loc='lower center',ncol=2,bbox_to_anchor=(.5,.04))
    fig.text(.5,.015,'Numerazione righe dal basso (Y crescente); stessa mappa nelle 128 scene.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,1))
    for extension in ('png','pdf'):
        fig.savefig(HERE/'maps'/f'layout.{extension}',dpi=160,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)


if __name__=='__main__':
    main()
