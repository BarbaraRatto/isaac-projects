"""Plot recorded evaluations; incomplete logs explicitly marked, no simulator."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from train_general_big_map_test2.missions import MissionSampler
OUT=Path(__file__).resolve().parent
MAP=json.loads((ROOT/'artifacts/map_metadata.json').read_text())
COLORS=['#d5d9dd','#a7deef','#e7dbab','#ecd0aa','#adb29f','#d1a784','#c5b5d9']
NAMES=['t1_asphalt','t2_slippery','t3_ramp','t4_fine_gravel','t5_large_rocks','t6_obstacles','t7_stairs']
REASONS={'goal':'Goal raggiunto','timeout':'Timeout','out_of_bounds':'Fuori mappa','fallen':'Caduta'}
SOURCES=[('200k','evaluation_20261025_104005_059237',2),('300k','evaluation_20261025_234206_938425',3)]
notes=[]
with PdfPages(OUT/'percorsi_200k_300k.pdf') as pdf:
 for label,folder,stage in SOURCES:
  src=ROOT/'results'/folder
  raw=(src/'trajectories.csv').read_text(); raw=raw[:raw.rfind('\n')+1]
  rows=list(csv.DictReader(raw.splitlines()))
  complete=(src/'summary.json').is_file()
  if complete:
   d=json.loads((src/'summary.json').read_text()); episodes=d['episodes']
   run=Path(d['checkpoint']).parent
   steps=json.loads((run/'summary.json').read_text())['transitions']
   assert steps==int(label[:-1])*1000,(steps,label)
  else:
   episodes=[];sampler=MissionSampler()
   for i in sorted({int(r['episode']) for r in rows}):
    m=sampler.evaluation_mission(0,i-1,20000,stage=stage)
    a=next(r for r in rows if int(r['episode'])==i)
    assert np.linalg.norm(np.array([float(a['x']),float(a['y'])])-m['start'][:2])<1e-4
    episodes.append(dict(episode=i,env_id=0,start_x=m['start'][0],start_y=m['start'][1],goal_x=m['goal'][0],goal_y=m['goal'][1],initial_yaw=m['yaw'],reason='pending'))
   notes.append(f'{label}: log incompleto; goal ricostruiti con seed 20000/fase 3, tutti gli start verificati. Checkpoint non ancora confermabile da summary.json; attribuzione 300k dal test richiesto.')
  fig,axes=plt.subplots(3,3,figsize=(16,16))
  for ax,e in zip(axes.flat,episodes):
   rr=sorted([r for r in rows if int(r['episode'])==e['episode'] and int(r['env_id'])==e['env_id']],key=lambda r:int(r['step']))
   xy=np.array([[float(r['x']),float(r['y'])] for r in rr])
   start=np.array([e['start_x'],e['start_y']]);goal=np.array([e['goal_x'],e['goal_y']])
   for ri,row in enumerate(MAP['grid']):
    for ci,name in enumerate(row):
     ax.add_patch(Rectangle((ci*6-3,ri*3-1.5),6,3,facecolor=COLORS[NAMES.index(name)],edgecolor='white',lw=.7,zorder=0))
   ax.add_patch(Rectangle((-3,-1.5),42,21,fill=False,edgecolor='#b41f24',lw=1.7))
   ax.plot(xy[:,0],xy[:,1],color='#2365bd',lw=1.8,zorder=3)
   ax.plot([*start[:1],goal[0]],[start[1],goal[1]],'k:',lw=.8,alpha=.5)
   ax.scatter(*start,color='black',s=42,zorder=6)
   ax.scatter(*goal,color='#158540',edgecolor='black',marker='*',s=165,zorder=6)
   ax.add_patch(Circle(goal,.55,fill=False,color='#158540',lw=1.3))
   ax.scatter(*xy[-1],color='#d34720',marker='x',s=50,zorder=7)
   ax.annotate('A',start,xytext=(5,6),textcoords='offset points',weight='bold')
   ax.annotate('B',goal,xytext=(5,6),textcoords='offset points',weight='bold')
   ax.arrow(*start,.6*np.cos(e['initial_yaw']),.6*np.sin(e['initial_yaw']),head_width=.12,color='black',length_includes_head=True,zorder=5)
   for j in np.linspace(0,max(0,len(xy)-4),min(3,len(xy)),dtype=int):
    k=min(len(xy)-1,j+3)
    ax.annotate('',xy=xy[k],xytext=xy[j],arrowprops=dict(arrowstyle='->',color='#2365bd'))
   bounds=np.vstack((xy,start,goal));lo=bounds.min(0)-.8;hi=bounds.max(0)+.8
   ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='X [m]',ylabel='Y [m]')
   outcome=REASONS.get(e['reason'],'Esito non ancora salvato')
   detail=(f"Percorso {e['path_m']:.1f} m | distanza finale {e['final_distance_m']:.2f} m" if complete else f"{len(xy)} campioni disponibili")
   ax.set_title(f"Episodio {e['episode']} — {outcome}\n{detail}",fontsize=10)
   ax.grid(alpha=.15)
  for ax in list(axes.flat)[len(episodes):]: ax.set_visible(False)
  suffix='' if complete else ' — DATI PROVVISORI'
  fig.suptitle(f'Policy {label} — fase {stage}{suffix}\nStart/goal casuali nella stessa cella di asfalto; distanza iniziale 1,5–3 m',fontsize=16,y=.99)
  handles=[Line2D([],[],color='black',marker='o',ls='',label='A: start; freccia: orientamento iniziale'),Line2D([],[],color='#158540',marker='*',ls='',markersize=13,label='B: goal; cerchio: raggio 0,55 m'),Line2D([],[],color='#2365bd',label='Percorso registrato'),Line2D([],[],color='#d34720',marker='x',ls='',label='Ultima posizione registrata')]
  fig.legend(handles=handles,loc='lower center',ncol=2,fontsize=10,bbox_to_anchor=(.5,.02))
  fig.text(.5,.008,'Assi in coordinate della mappa; scale diverse per episodio. Ultimo campione precedente all’ultimo passo, non posa terminale esatta.',ha='center',fontsize=9)
  fig.tight_layout(rect=(0,.075,1,.95), h_pad=3.5, w_pad=2)
  fig.savefig(OUT/f'percorsi_{label}.png',dpi=150)
  fig.savefig(OUT/f'percorsi_{label}.pdf')
  pdf.savefig(fig);plt.close(fig)
  notes.append(f'{label}: fonte {src}; fase {stage}; completo={complete}.')
(OUT/'NOTE.md').write_text('# Percorsi dei test 200k e 300k\n\n'+'\n\n'.join(notes)+'\n\nLe missioni differiscono fra fase 2 e fase 3: non è un confronto a parità di start/goal/orientamento. Il punto arancione è l’ultimo campione pre-step, fino a 0,2 s prima della posa terminale.\n')
print(OUT/'percorsi_200k_300k.pdf')
