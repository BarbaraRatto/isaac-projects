"""Offline comparison of matching evaluated missions, with start and goal markers."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
COLORS=['#1359bc','#bd4d08','#991b96']
TERRAINS={'t1_asphalt':('#c4c4c4','Asfalto'), 't2_slippery':('#a1dcec','Ghiaccio'),
          't3_ramp':('#ded39a','Rampa'), 't4_fine_gravel':('#e6d0b5','Ghiaia'),
          't5_large_rocks':('#a7a899','Rocce'), 't6_obstacles':('#cfae8f','Ostacoli'),
          't7_stairs':('#c8b9d8','Scale')}
REASONS={'fallen':'Caduta','out_of_bounds':'Fuori mappa','timeout':'Timeout','goal':'Goal'}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('evaluations',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
 args=p.parse_args()
 if len(args.evaluations)>3: p.error('At most three policies')
 args.output.mkdir(parents=True,exist_ok=True)
 meta=json.loads((HERE/'artifacts/map_metadata.json').read_text())
 runs=[];rows=[]
 for folder in args.evaluations:
  d=json.loads((folder/'summary.json').read_text())
  label=Path(d['checkpoint']).stem.removeprefix('policy_')
  with (folder/'trajectories.csv').open() as f: traj=list(csv.DictReader(f))
  missions={}
  for e in d['episodes']:
   key=(e['env_id'],e['episode'])
   samples=sorted([x for x in traj if int(x['env_id'])==key[0] and int(x['episode'])==key[1]],key=lambda x:int(x['step']))
   assert len(samples)==e['steps'],(folder,key)
   xy=np.array([[float(x['x']),float(x['y'])] for x in samples])
   times=np.array([int(x['step'])*.2 for x in samples])
   dist=np.linalg.norm(xy-[e['goal_x'],e['goal_y']],axis=1)
   minimum=float(min(dist.min(),e['final_distance_m']))
   row=dict(policy=label,episode=e['episode'],env_id=e['env_id'],band=e['band'],reason=e['reason'],
            initial_distance_m=e['initial_distance_m'],minimum_sampled_distance_m=minimum,
            final_distance_m=e['final_distance_m'],path_m=e['path_m'],seconds=e['steps']*.2,energy_j=e['energy_j'])
   rows.append(row);missions[key]=(e,xy,times,dist,row)
  runs.append(dict(label=label,missions=missions,summary=d['summary'],source=str(folder)))
 keys=sorted(runs[0]['missions'])
 for run in runs[1:]:
  assert sorted(run['missions'])==keys
  for key in keys:
   a=runs[0]['missions'][key][0];b=run['missions'][key][0]
   assert all(np.isclose(a[x],b[x]) for x in ('start_x','start_y','goal_x','goal_y','initial_yaw','deadline_s'))
 def draw_map(ax,key,small=False):
  allxy=[]
  for r,row in enumerate(meta['grid']):
   for c,name in enumerate(row):
    ax.add_patch(Rectangle((c*6-3,r*3-1.5),6,3,facecolor=TERRAINS[name][0],edgecolor='white',lw=.6,alpha=.65))
  for idx,run in enumerate(runs):
   e,xy,times,dist,_=run['missions'][key];allxy.append(xy)
   ax.plot(xy[:,0],xy[:,1],color=COLORS[idx],lw=1.4,label=run['label'])
   ax.scatter(*xy[-1],marker='x',color=COLORS[idx],s=35,zorder=6)
   if not small:
    for j in np.linspace(0,len(xy)-2,min(5,len(xy)-1),dtype=int):
     end=min(j+3,len(xy)-1)
     ax.annotate('',xy=xy[end],xytext=xy[j],arrowprops=dict(arrowstyle='->',color=COLORS[idx],lw=1))
  e=runs[0]['missions'][key][0];start=np.array([e['start_x'],e['start_y']]);goal=np.array([e['goal_x'],e['goal_y']])
  ax.plot([*start[:1],*goal[:1]],[start[1],goal[1]],'k:',alpha=.45,lw=1)
  ax.scatter(*start,c='black',marker='o',s=38,zorder=8)
  ax.scatter(*goal,c='#007d25',edgecolors='black',marker='*',s=150,zorder=9)
  ax.add_patch(Circle(goal,.55,fill=False,edgecolor='#007d25',lw=1.3,zorder=7))
  ax.annotate('A',start,xytext=(5,5),textcoords='offset points',weight='bold')
  ax.annotate('B',goal,xytext=(5,5),textcoords='offset points',weight='bold',color='#007d25')
  if not small:
   ax.arrow(*start,.8*np.cos(e['initial_yaw']),.8*np.sin(e['initial_yaw']),head_width=.18,color='black',zorder=7,length_includes_head=True)
  pts=np.vstack([*allxy,start[None],goal[None]]);lo=pts.min(0)-1;hi=pts.max(0)+1
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='X (m)',ylabel='Y (m)',aspect='equal')
  bands=['3–6 m','6–12 m','12–20 m']
  ax.set_title(f'Missione {key[1]} — {bands[e["band"]]}',fontsize=10)
 def draw_distance(ax,key):
  for idx,run in enumerate(runs):
   e,xy,times,dist,_=run['missions'][key]
   ax.plot(np.r_[times,e['steps']*.2],np.r_[dist,e['final_distance_m']],color=COLORS[idx],label=f'{run["label"]}: {REASONS[e["reason"]]}')
  ax.axhline(.55,color='green',ls='--',label='Soglia goal: 0,55 m')
  ax.set(xlabel='Tempo (s)',ylabel='Distanza dal goal (m)',title='Avvicinamento al goal');ax.grid(alpha=.2);ax.legend(fontsize=9)
 fig,axes=plt.subplots(3,3,figsize=(16,13),layout='constrained')
 for ax,key in zip(axes.flat,keys): draw_map(ax,key,True)
 for ax in list(axes.flat)[len(keys):]:ax.axis('off')
 fig.suptitle('Stesse missioni, tre policy — A: partenza; B: goal (raggio 0,55 m)\nBlu: 53.248 — Arancio: 253.952 — Viola: 450.560; ×: ultimo campione prima della fine',fontsize=13)
 fig.savefig(args.output/'paths_overview.png',dpi=160);fig.savefig(args.output/'paths_overview.pdf');plt.close(fig)
 with PdfPages(args.output/'missions_detail.pdf') as pdf:
  for key in keys:
   fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained')
   draw_map(axes[0],key);draw_distance(axes[1],key)
   text=[]
   for run in runs:
    e=run['missions'][key][0]
    text.append(f'{run["label"]}: {REASONS[e["reason"]]}, {e["steps"]*.2:.1f} s, percorso {e["path_m"]:.1f} m, distanza finale {e["final_distance_m"]:.2f} m, {e["energy_j"]:.0f} J')
   fig.suptitle('\n'.join(text),fontsize=11)
   legend=[Patch(facecolor=c,label=name) for c,name in TERRAINS.values()]
   fig.legend(handles=legend,loc='outside lower center',ncols=7,fontsize=9)
   pdf.savefig(fig)
   if key[1]==1:fig.savefig(args.output/'mission_1.png',dpi=150)
   plt.close(fig)
 with (args.output/'comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report=['# Confronto policy su missioni identiche','',
 'A = partenza; B = goal; cerchio verde = raggio di successo 0,55 m.',
 'Le curve sulla mappa terminano all’ultimo campione pre-step: manca la posizione dopo l’ultimo passo (0,2 s). La distanza terminale e le metriche episodiche provengono invece dal riepilogo esatto.',
 'Il minimo riportato è quello fra campioni registrati e distanza terminale, non un minimo continuo.',
 'I colori di fondo indicano le celle/materiali, non la quota o gli ostacoli interni. La linea A–B è un riferimento geometrico, non un percorso certificato attraversabile.',
 '', '| Policy | Successi | Cadute | Fuori mappa | Timeout |','|---|---:|---:|---:|---:|']
 for run in runs:
  s=run['summary'];o=s['outcomes'];report.append(f'| {run["label"]} | {s["successes"]}/{s["episodes"]} | {o["fallen"]} | {o["out_of_bounds"]} | {o["timeout"]} |')
 report+=['','Le missioni 1, 4, 7 appartengono alla fascia 3–6 m allenata. Le altre due fasce non erano state raggiunte dal curriculum.',
 'Non confrontare l’efficienza energetica dei fallimenti come se fossero missioni completate.', '', '## Sorgenti']
 report += [f'- {r["label"]}: {r["source"]}' for r in runs]
 (args.output/'REPORT.md').write_text('\n'.join(report)+'\n')
 print('Saved',args.output)
 for row in rows:
  if row['band']==0: print(json.dumps(row))

if __name__=='__main__':main()
