"""Offline paired trajectory plots; no simulator or model changes."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT/'results/goal_normalization_20261029_122904_RevSuP'
COLORS = ['#1465b0', '#d66000']
LABELS = ['Originale 300k', '300k con statistiche goal 100k']
TERRAINS = {'t1_asphalt': ('#d5d9dd','Asfalto'), 't2_slippery': ('#a7deef','Ghiaccio'),
            't3_ramp': ('#e7dbab','Rampe'), 't4_fine_gravel': ('#ecd0aa','Ghiaia'),
            't5_large_rocks': ('#adb29f','Rocce'), 't6_obstacles': ('#d1a784','Ostacoli'),
            't7_stairs': ('#c5b5d9','Scale')}
REASONS = {'goal':'Goal', 'timeout':'Timeout', 'fallen':'Caduta', 'out_of_bounds':'Fuori mappa', 'alignment_timeout':'Allineamento fallito'}
meta = json.loads((ROOT/'artifacts/map_metadata.json').read_text())
data = []
for folder in ('original_300k', 'goal_norm_100k'):
    doc = json.loads((SOURCE/folder/'summary.json').read_text())
    with (SOURCE/folder/'trajectories.csv').open() as f:
        rows = list(csv.DictReader(f))
    episodes = {}
    for e in doc['episodes']:
        r = sorted([r for r in rows if int(r['episode'])==e['episode'] and int(r['env_id'])==e['env_id']], key=lambda r:int(r['step']))
        assert len(r)==e['steps']
        assert [int(r['step']) for r in r]==list(range(e['steps']))
        xy = np.array([[float(r['x']),float(r['y'])] for r in r]+[[e['final_x'],e['final_y']]])
        np.testing.assert_allclose(np.linalg.norm(xy[-1]-[e['goal_x'],e['goal_y']]),e['final_distance_m'],atol=1e-4)
        np.testing.assert_allclose([[float(r['goal_x']),float(r['goal_y'])] for r in r], np.tile([e['goal_x'],e['goal_y']],(len(r),1)),atol=1e-4)
        episodes[e['episode']] = (e,xy)
    assert len(episodes)==9
    data.append(episodes)
for n in range(1,10):
    a,b = data[0][n][0],data[1][n][0]
    for k in ('start_id','goal_id','layout','band'): assert a[k]==b[k]
    for k in ('start_x','start_y','goal_x','goal_y','initial_yaw'): np.testing.assert_allclose(a[k],b[k],atol=1e-4,rtol=0)
config=json.loads((Path(doc['checkpoint']).parent/'config.json').read_text())['signature']['navigation']
DT=config['physics_dt']*config['physics_steps_per_action']; RADIUS=config['goal_radius_m']
handles=[Line2D([],[],color=c,lw=2,label=l) for c,l in zip(COLORS,LABELS)]
handles += [Line2D([],[],color='black',marker='o',ls='',label='A: start'),
            Line2D([],[],color='green',marker='*',markersize=12,ls='',label='B: goal (cerchio 0,55 m)'),
            Line2D([],[],color='black',marker='x',ls='',label='Fine episodio'),
            Line2D([],[],color='gray',ls='--',label='Allineamento iniziale')]
terrain_handles=[Patch(facecolor=c,label=l) for c,l in TERRAINS.values()]

def draw(ax,n,variants=(0,1)):
    e=data[0][n][0]
    for ri,row in enumerate(meta['grid']):
        for ci,terrain in enumerate(row):
            ax.add_patch(Rectangle((ci*6-3,ri*3-1.5),6,3,facecolor=TERRAINS[terrain][0],edgecolor='white',lw=.7))
    start=np.array([e['start_x'],e['start_y']]); goal=np.array([e['goal_x'],e['goal_y']])
    for i in variants:
        record,xy=data[i][n]; k=record['alignment_steps']
        ax.plot(xy[:k+1,0],xy[:k+1,1],color=COLORS[i],ls='--',lw=1.5)
        ax.plot(xy[k:,0],xy[k:,1],color=COLORS[i],lw=1.6)
        ax.scatter(*xy[-1],marker='x',color=COLORS[i],s=65,zorder=8)
        for j in np.linspace(k,max(k,len(xy)-4),4,dtype=int):
            ax.annotate('',xy=xy[min(j+3,len(xy)-1)],xytext=xy[j],arrowprops=dict(arrowstyle='->',color=COLORS[i],lw=1))
    ax.plot([*start[:1],*goal[:1]],[start[1],goal[1]],ls=':',color='black',alpha=.4)
    ax.scatter(*start,color='black',s=30,zorder=7); ax.annotate('A',start,xytext=(5,6),textcoords='offset points',weight='bold')
    ax.scatter(*goal,marker='*',color='green',edgecolor='black',s=130,zorder=7); ax.annotate('B',goal,xytext=(6,6),textcoords='offset points',weight='bold')
    ax.add_patch(Circle(goal,RADIUS,fill=False,color='green',lw=1.2))
    points=np.vstack((data[0][n][1],data[1][n][1],start,goal)); low=points.min(0)-.7; high=points.max(0)+.7
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal',xlabel='X [m]',ylabel='Y [m]'); ax.grid(alpha=.15)
    title=f'Episodio {n}'
    for i in variants:
        record=data[i][n][0]
        title+=f"\n{'Originale' if i==0 else 'Goal norm 100k'}: {REASONS[record['reason']]} | {record['path_m']:.1f} m | {record['energy_j']:.0f} J"
    ax.set_title(title,fontsize=10)

fig,axes=plt.subplots(3,3,figsize=(18,18))
for n,ax in enumerate(axes.flat,1): draw(ax,n)
fig.suptitle('Stesse nove missioni — confronto dei percorsi\nBlu: originale 300k | Arancione: solo normalizzazione goal da 100k',fontsize=17)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3)
fig.legend(handles=terrain_handles,loc='lower center',bbox_to_anchor=(.5,.02),ncol=7,fontsize=9)
fig.tight_layout(rect=(0,.10,1,.95),h_pad=6)
fig.savefig(OUT/'confronto_percorsi.pdf'); fig.savefig(OUT/'confronto_percorsi.png',dpi=130); plt.close(fig)
with PdfPages(OUT/'episodi_affiancati.pdf') as pdf:
    for n in range(1,10):
        fig,axes=plt.subplots(1,3,figsize=(18,7))
        draw(axes[0],n,(0,)); draw(axes[1],n,(1,))
        for i in range(2):
            e,xy=data[i][n]
            distances=np.linalg.norm(xy-[e['goal_x'],e['goal_y']],axis=1)
            axes[2].plot(np.arange(len(xy))*DT,distances,color=COLORS[i],label=LABELS[i])
        axes[2].axhline(RADIUS,color='green',ls='--'); axes[2].set(xlabel='Tempo totale [s]',ylabel='Distanza dal goal [m]',title='Avvicinamento al goal'); axes[2].grid(alpha=.2); axes[2].legend(fontsize=8)
        fig.legend(handles=terrain_handles,loc='lower center',ncol=7)
        fig.suptitle('Stessa scala spaziale nelle due varianti; X = posizione finale',fontsize=13)
        fig.tight_layout(rect=(0,.09,1,.94)); pdf.savefig(fig); plt.close(fig)
common=[n for n in range(1,10) if all(data[i][n][0]['success'] for i in range(2))]
lines=['# Analisi del confronto in simulazione','',f'Missioni riuscite per entrambe le varianti: {common}.','',
       '| Metrica | Originale | Solo goal 100k |','|---|---:|---:|']
for i in range(2):
    counts={reason:sum(data[i][n][0]['reason']==reason for n in range(1,10)) for reason in REASONS}
    print(LABELS[i],counts)
lines.append(f"| Successi su tutte le missioni | {sum(data[0][n][0]['success'] for n in range(1,10))}/9 | {sum(data[1][n][0]['success'] for n in range(1,10))}/9 |")
for field,label,scale in [('path_m','Percorso medio, successi comuni (m)',1),('energy_j','Energia media, successi comuni (J)',1),('steps','Durata media, successi comuni (s)',DT)]:
    means=[np.mean([data[i][n][0][field]*scale for n in common]) for i in range(2)]
    lines.append(f'| {label} | {means[0]:.2f} | {means[1]:.2f} |')
    print(label,means,'change%',100*(means[1]/means[0]-1))
lines+=['','Totali comprensivi di allineamento. Non confrontare il minor consumo di una caduta con quello di una missione completata.',
        'I grafici includono la posizione terminale esatta e verificano start, goal e missioni corrispondenti.',
        'Un solo confronto per missione: risultati descrittivi, non prova statistica né identificazione della causa storica del training.','',
        (SOURCE/'REPORT.md').read_text()]
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('Saved:',OUT)
