"""Offline plots of the completed test4 evaluation; never launches Isaac Sim."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'results/evaluation_20261027_130913_760448'
doc=json.loads((SOURCE/'summary.json').read_text())
with (SOURCE/'episodes.csv').open() as stream: episode_rows=list(csv.DictReader(stream))
transitions={int(row['transitions']) for row in episode_rows}
assert len(transitions)==1
transitions=transitions.pop()
assert transitions==300000
run_config=json.loads((Path(doc['checkpoint']).parent/'config.json').read_text())
nav=run_config['signature']['navigation']
DT=nav['physics_dt']*nav['physics_steps_per_action']
RADIUS=nav['goal_radius_m']
stage=doc['summary']['stage']
summary=doc['summary']
assert len(doc['episodes'])==9
assert sum(e['success'] for e in doc['episodes'])==summary['successes']

meta=json.loads((ROOT/'artifacts/map_metadata.json').read_text())
with (SOURCE/'trajectories.csv').open() as f: rows=list(csv.DictReader(f))
TERRAINS={'t1_asphalt':('#d5d9dd','Asfalto'),'t2_slippery':('#a7deef','Ghiaccio'),
          't3_ramp':('#e7dbab','Rampe'),'t4_fine_gravel':('#ecd0aa','Ghiaia'),
          't5_large_rocks':('#adb29f','Rocce'),'t6_obstacles':('#d1a784','Ostacoli'),
          't7_stairs':('#c5b5d9','Scale')}
REASONS={'goal':'Goal raggiunto','timeout':'Timeout','fallen':'Caduta','out_of_bounds':'Fuori mappa','alignment_timeout':'Allineamento fallito'}
BLUE='#165ca8';ORANGE='#df7900';GREEN='#14823b';RED='#bf2541'
missions=[]
for e in doc['episodes']:
 r=sorted([x for x in rows if int(x['env_id'])==e['env_id'] and int(x['episode'])==e['episode']],key=lambda x:int(x['step']))
 assert len(r)==e['steps']
 assert [int(x['step']) for x in r]==list(range(e['steps']))
 assert sum(x['phase']=='alignment' for x in r)==e['alignment_steps']
 assert sum(x['phase']=='navigation' for x in r)==e['navigation_steps']
 np.testing.assert_allclose([[float(x['goal_x']),float(x['goal_y'])] for x in r],np.tile([e['goal_x'],e['goal_y']],(len(r),1)),atol=1e-4)
 xy=np.array([[float(x['x']),float(x['y'])] for x in r]+[[e['final_x'],e['final_y']]])
 np.testing.assert_allclose(np.linalg.norm(xy[-1]-[e['goal_x'],e['goal_y']]),e['final_distance_m'],atol=1e-4)
 missions.append((e,xy))

def draw(ax,mission,small=True):
 e,xy=mission
 for ri,row in enumerate(meta['grid']):
  for ci,name in enumerate(row):
   ax.add_patch(Rectangle((ci*6-3,ri*3-1.5),6,3,facecolor=TERRAINS[name][0],edgecolor='white',lw=.7,zorder=0))
 ax.add_patch(Rectangle((-3,-1.5),42,21,fill=False,edgecolor=RED,lw=1.2))
 start=np.array([e['start_x'],e['start_y']]);goal=np.array([e['goal_x'],e['goal_y']])
 ax.plot([start[0],goal[0]],[start[1],goal[1]],color='black',ls=':',lw=.9,alpha=.5)
 n=e['alignment_steps']
 ax.plot(xy[:n+1,0],xy[:n+1,1],color=ORANGE,lw=2.4,zorder=4)
 if e['navigation_steps']:
  ax.plot(xy[n:,0],xy[n:,1],color=BLUE,lw=1.8,zorder=3)
  ax.scatter(*xy[n],color=ORANGE,marker='D',s=22,zorder=6)
  for j in np.linspace(n,max(n,len(xy)-4),min(4,e['navigation_steps']),dtype=int):
   k=min(len(xy)-1,j+3)
   ax.annotate('',xy=xy[k],xytext=xy[j],arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.1))
 ax.scatter(*start,color='black',s=38,zorder=7)
 ax.scatter(*goal,color=GREEN,edgecolor='black',marker='*',s=160,zorder=7)
 ax.add_patch(Circle(goal,RADIUS,fill=False,color=GREEN,lw=1.3))
 ax.scatter(*xy[-1],color=RED,marker='x',s=65,zorder=8)
 ax.annotate('A',start,xytext=(6,7),textcoords='offset points',weight='bold')
 ax.annotate('B',goal,xytext=(6,7),textcoords='offset points',weight='bold',color=GREEN)
 ax.arrow(*start,.65*np.cos(e['initial_yaw']),.65*np.sin(e['initial_yaw']),head_width=.14,color='black',length_includes_head=True,zorder=5)
 points=np.vstack((xy,start,goal));lo=points.min(0)-.9;hi=points.max(0)+.9
 ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='X [m]',ylabel='Y [m]')
 ax.grid(alpha=.15)
 title=f"Episodio {e['episode']} — {REASONS[e['reason']]}\nDistanza iniziale {e['initial_distance_m']:.2f} m | percorso {e['path_m']:.2f} m"
 if not small:title+=f"\nEnergia totale {e['energy_j']:.0f} J | tempo {e['steps']*DT:.1f} s | distanza finale {e['final_distance_m']:.2f} m"
 ax.set_title(title,fontsize=10 if small else 13,pad=12)

handles=[Line2D([],[],color='black',marker='o',ls='',label='A: start; freccia nera: orientamento iniziale'),
         Line2D([],[],color=GREEN,marker='*',ls='',markersize=12,label='B: goal; cerchio: raggio 0,55 m'),
         Line2D([],[],color=ORANGE,lw=2,label='Allineamento automatico'),
         Line2D([],[],color=BLUE,lw=2,label='Navigazione RL'),
         Line2D([],[],color=ORANGE,marker='D',ls='',label='Inizio controllo RL'),
         Line2D([],[],color=RED,marker='x',ls='',label='Posizione finale esatta')]
terrain_handles=[Patch(facecolor=color,label=name) for color,name in TERRAINS.values()]
fig,axes=plt.subplots(3,3,figsize=(18,18))
for ax,m in zip(axes.flat,missions):draw(ax,m)
fig.suptitle(f"Test 4 — policy a {transitions:,} transizioni — fase {stage}\n{summary['episodes']} episodi: {summary['successes']} goal raggiunti, {summary['outcomes']['fallen']} cadute, {summary['outcomes']['timeout']} timeout | Allineamento riuscito: {sum(e['alignment_success'] for e,_ in missions)}/9",fontsize=17,y=.985)
fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.052),fontsize=10)
fig.legend(handles=terrain_handles,loc='lower center',ncol=7,bbox_to_anchor=(.5,.027),fontsize=9)
fig.text(.5,.012,'Coordinate della mappa in metri; scale diverse per episodio. Percorso ed energia totali includono l’allineamento.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.105,1,.945),h_pad=4,w_pad=2.5)
fig.savefig(OUT/'percorsi_test4_300k.pdf')
fig.savefig(OUT/'percorsi_test4_300k.png',dpi=150)
plt.close(fig)
with PdfPages(OUT/'episodi_dettaglio.pdf') as pdf:
 for m in missions:
  e,xy=m
  fig,(ax,distax)=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1.4,1]})
  draw(ax,m,small=False)
  distances=np.linalg.norm(xy-[e['goal_x'],e['goal_y']],axis=1)
  times=np.arange(len(xy))*DT
  distax.axvspan(0,e['alignment_steps']*DT,color=ORANGE,alpha=.15,label='Allineamento')
  distax.plot(times,distances,color=BLUE)
  distax.axhline(RADIUS,color=GREEN,ls='--',label='Raggio goal')
  distax.set(xlabel='Tempo dalla partenza [s]',ylabel='Distanza dal goal [m]',title='Avvicinamento al goal',ylim=(0,max(distances)*1.08))
  distax.grid(alpha=.25);distax.legend()
  fig.suptitle(f"Test 4 — episodio {e['episode']} — allineamento {e['alignment_steps']*DT:.1f} s / {e['alignment_energy_j']:.0f} J; navigazione {e['navigation_steps']*DT:.1f} s / {e['navigation_energy_j']:.0f} J",fontsize=13)
  fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.035),fontsize=9)
  fig.legend(handles=terrain_handles,loc='lower center',ncol=7,bbox_to_anchor=(.5,.003),fontsize=8)
  fig.tight_layout(rect=(0,.13,1,.94),w_pad=3)
  pdf.savefig(fig);plt.close(fig)
fields=['episode','reason','start_terrain','goal_terrain','start_x','start_y','goal_x','goal_y',
        'initial_distance_m','path_m','path_over_initial_distance','final_distance_m','seconds','deadline_s',
        'energy_j','alignment_energy_j','navigation_energy_j','alignment_steps','navigation_steps']
with (OUT/'riepilogo_episodi.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for e,_ in missions:
  row=dict(e, seconds=e['steps']*DT, path_over_initial_distance=e['path_m']/e['initial_distance_m'])
  w.writerow({k:row[k] for k in fields})
table=['| Episodio | Esito | A–B diretto (m) | Percorso (m) | Distanza finale (m) | Tempo (s) | Energia (J) |',
       '|---|---|---:|---:|---:|---:|---:|']
for e,_ in missions:
 table.append(f"| {e['episode']} | {REASONS[e['reason']]} | {e['initial_distance_m']:.2f} | {e['path_m']:.2f} | {e['final_distance_m']:.2f} | {e['steps']*DT:.1f} | {e['energy_j']:.0f} |")
text=['# Percorsi test 4 — 300.000 transizioni','',f"Fonte: `{SOURCE}`",f"Checkpoint: `{doc['checkpoint']}`",'',
      f"Fase {stage}: {summary['stage_name']}. Successi: {summary['successes']}/{summary['episodes']} ({summary['success_rate']:.1%}); 3 cadute e 2 timeout. Allineamento completato in tutti i 9 episodi.", '',*table,'',
      'Tempo, percorso ed energia includono l’allineamento automatico. A–B diretto è la distanza euclidea iniziale, non un percorso ottimo calcolato sul terreno. Il raggio di raggiungimento è 0,55 m: non occorre arrivare al centro del goal.', '',
      '## Lettura dei risultati', '',
      'I successi sono negli episodi 1, 4, 7 e 8. L’episodio 8 completa una missione con distanza iniziale di 3,53 m percorrendo 3,53 m in 6,2 s e consumando circa 699 J. Negli altri tre successi il percorso è più lungo: 10,13 m, 16,75 m e 13,33 m, a fronte di distanze dirette di 4,86 m, 5,14 m e 4,68 m.', '',
      'I timeout 2 e 9 terminano più lontano dal goal rispetto alla partenza, dopo percorsi rispettivamente di 22,80 m e 19,57 m. Non si tratta semplicemente di un robot lento che si arresta poco prima del goal. L’allineamento automatico riesce sempre; i problemi osservati avvengono durante la navigazione RL. Il risultato resta insufficiente per una navigazione affidabile. Nove episodi descrivono queste missioni e non bastano a stimare con precisione la prestazione generale o a isolare l’effetto della PCA rispetto alla reward.', '',
      f"Solo successi: energia media {summary['mean_success_energy_j']:.0f} J; percorso medio {summary['mean_success_path_m']:.2f} m; tempo medio {summary['mean_success_seconds']:.2f} s.", '',
      '## Grafici e verifiche', '',
      'Start A nero, goal B verde con raggio 0,55 m, orientamento iniziale indicato dalla freccia nera, allineamento arancione, navigazione blu, inizio RL a rombo e posa terminale esatta a croce rossa. Il segmento tratteggiato A–B è un riferimento geometrico, non una traiettoria pianificata. Le scale cambiano tra episodi. I colori del terreno derivano dalla mappa congelata.', '',
      '`percorsi_test4_300k.pdf` e `.png`: panoramica dei nove percorsi. `episodi_dettaglio.pdf`: una pagina per episodio, con percorso ingrandito e distanza dal goal nel tempo. `riepilogo_episodi.csv`: dati della tabella, coordinate, materiali, tempi/energia separati per allineamento e navigazione.', '',
      'La posa terminale è aggiunta dai record degli episodi; i campioni di traiettoria sono pre-step. Controllati numero e ordine dei campioni, fasi allineamento/navigazione, goal registrati e distanza finale ricostruita. Nessuna simulazione avviata.', '',
      'Riproduzione offline (Python di sistema, senza pacchetti utente, per mantenere compatibili NumPy e Matplotlib):', '',
      '```bash', 'PYTHONNOUSERSITE=1 /usr/bin/python3 spot_rl_energy_navigation/train_general_big_map_test4/analysis/paths_20261027/plot_paths.py', '```']
(OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
print('\n'.join(table))
print('Generated plots for',len(missions),'episodes in',OUT)
