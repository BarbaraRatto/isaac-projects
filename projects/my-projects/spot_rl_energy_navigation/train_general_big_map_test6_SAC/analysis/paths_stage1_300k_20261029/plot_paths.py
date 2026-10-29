"""Offline plots of the completed test6 SAC phase 1 evaluation; never launches Isaac Sim."""
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
SOURCE=ROOT/'results/evaluation_20261029_110049_426900'
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
assert stage==1
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
fig.suptitle(f"Test 6 SAC — policy a {transitions:,} transizioni — fase {stage}\n{summary['episodes']} episodi: {summary['successes']} goal raggiunti, {summary['outcomes']['fallen']} cadute, {summary['outcomes']['timeout']} timeout, {summary['outcomes']['out_of_bounds']} fuori mappa | Allineamento riuscito: {sum(e['alignment_success'] for e,_ in missions)}/9",fontsize=17,y=.985)
fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.052),fontsize=10)
fig.legend(handles=terrain_handles,loc='lower center',ncol=7,bbox_to_anchor=(.5,.027),fontsize=9)
fig.text(.5,.012,'Coordinate della mappa in metri; scale diverse per episodio. Percorso ed energia totali includono l’allineamento.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.105,1,.945),h_pad=4,w_pad=2.5)
fig.savefig(OUT/'percorsi_test6_SAC_fase1_300k.pdf')
fig.savefig(OUT/'percorsi_test6_SAC_fase1_300k.png',dpi=150)
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
  fig.suptitle(f"Test 6 SAC — episodio {e['episode']} — allineamento {e['alignment_steps']*DT:.1f} s / {e['alignment_energy_j']:.0f} J; navigazione {e['navigation_steps']*DT:.1f} s / {e['navigation_energy_j']:.0f} J",fontsize=13)
  fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.035),fontsize=9)
  fig.legend(handles=terrain_handles,loc='lower center',ncol=7,bbox_to_anchor=(.5,.003),fontsize=8)
  fig.tight_layout(rect=(0,.13,1,.94),w_pad=3)
  pdf.savefig(fig);plt.close(fig)
fields=['episode','reason','start_terrain','goal_terrain','start_x','start_y','goal_x','goal_y',
        'initial_distance_m','path_m','path_over_initial_distance','final_distance_m','seconds','deadline_s',
        'energy_j','alignment_energy_j','navigation_energy_j','alignment_steps','navigation_steps','fall_cause','max_tilt_deg','max_tilt_duration_s','max_body_contact_duration_s','legacy_fall_steps']
with (OUT/'riepilogo_episodi.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for e,_ in missions:
  row=dict(e, seconds=e['steps']*DT, path_over_initial_distance=e['path_m']/e['initial_distance_m'])
  w.writerow({k:row[k] for k in fields})
table=['| Episodio | Esito | A–B diretto (m) | Percorso (m) | Distanza finale (m) | Tempo (s) | Energia (J) |',
       '|---|---|---:|---:|---:|---:|---:|']
for e,_ in missions:
 table.append(f"| {e['episode']} | {REASONS[e['reason']]} | {e['initial_distance_m']:.2f} | {e['path_m']:.2f} | {e['final_distance_m']:.2f} | {e['steps']*DT:.1f} | {e['energy_j']:.0f} |")

with (SOURCE/'fall_diagnostics.csv').open() as stream:
 diagnostics=list(csv.DictReader(stream))
for e,xy in missions:
 ds=sorted([r for r in diagnostics if int(r['episode'])==e['episode'] and int(r['env_id'])==e['env_id']],key=lambda r:int(r['step']))
 assert [int(r['step']) for r in ds]==list(range(1,e['steps']+1))
 np.testing.assert_allclose([[float(r['x']),float(r['y'])] for r in ds],xy[1:],atol=1e-4)
 np.testing.assert_allclose([float(r['time_s']) for r in ds],np.arange(1,e['steps']+1)*DT,atol=1e-5)

# Distances and paths are distinct quantities; plot both with explicit units.
fig,axes=plt.subplots(1,2,figsize=(13,5))
numbers=np.array([e['episode'] for e,_ in missions])
axes[0].bar(numbers-.18,[e['initial_distance_m'] for e,_ in missions],width=.36,color='#777777',label='Distanza iniziale A–B')
axes[0].bar(numbers+.18,[e['path_m'] for e,_ in missions],width=.36,color=[GREEN if e['success'] else RED for e,_ in missions],label='Percorso totale')
axes[0].set(xlabel='Episodio',ylabel='Metri',title='Distanza iniziale e strada percorsa',xticks=numbers)
axes[0].legend()
axes[1].bar(numbers,[e['energy_j'] for e,_ in missions],color=[GREEN if e['success'] else RED for e,_ in missions])
axes[1].set(xlabel='Episodio',ylabel='Energia meccanica totale [J]',title='Energia per episodio',xticks=numbers)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('Test 6 SAC — fase 1 — verde: goal, rosso: timeout')
fig.text(.5,.01,'Totali comprensivi di allineamento. Gli episodi falliti non completano la stessa task dei successi.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94))
fig.savefig(OUT/'riepilogo_metriche.pdf');plt.close(fig)

minimum=[]
for e,xy in missions:
 minimum.append((e['episode'],float(np.linalg.norm(xy-[e['goal_x'],e['goal_y']],axis=1).min())))
report=[
 '# Test 6 SAC — fase 1 — 300.000 transizioni','',
 f"Fonte: `{SOURCE}`",f"Checkpoint: `{doc['checkpoint']}`",'',
 f"Successi: {summary['successes']}/9 ({100*summary['success_rate']:.1f}%). Un timeout, nessuna caduta o uscita dalla mappa. Allineamento riuscito 9/9. Fase {stage}: {summary['stage_name']}.",'',
 *table,'',
 'Distanze in metri, energia meccanica in joule. Tempo, percorso ed energia includono l’allineamento. Il raggio di successo è 0,55 m. A–B è la distanza euclidea, non una traiettoria ottima sul terreno.','',
 '## Interpretazione','',
 'Otto missioni su nove vengono completate. Resta un problema di percorsi lunghi: nei successi le traiettorie misurano da 8,03 a 15,51 m, a fronte di distanze iniziali di 3,47–5,99 m. Raggiungere il goal non dimostra di aver imparato una navigazione energeticamente efficiente.','',
 f"Nei soli successi: percorso medio {summary['mean_success_path_m']:.2f} m, tempo medio {summary['mean_success_seconds']:.2f} s, energia media {summary['mean_success_energy_j']:.0f} J.",'',
 'L’episodio 4 termina per timeout dopo 35,6 s e 22,22 m percorsi, a 0,79 m dal centro del goal: circa 0,24 m fuori dal raggio di successo. È vicino al goal alla scadenza, ma la lunghezza del percorso segnala che aumentare il timeout da solo non risolverebbe l’inefficienza.','',
 'Questo test usa la fase 1; il precedente test SAC a 100.000 transizioni usava la fase 0. Le due percentuali non sono un confronto a difficoltà uguale. Le nove missioni non bastano a stimare con precisione il successo generale.','',
 '## Grafici e verifiche','',
 '- `percorsi_test6_SAC_fase1_300k.pdf` / `.png`: nove percorsi, start A nero, goal B verde, allineamento arancione, navigazione blu e posa finale a croce rossa.',
 '- `episodi_dettaglio.pdf`: una pagina per episodio, con percorso e distanza dal goal nel tempo.',
 '- `riepilogo_metriche.pdf`: distanza iniziale, percorso ed energia.',
 '- `riepilogo_episodi.csv`: dati numerici e coordinate.','',
 'Scale diverse per episodio. La posa terminale è aggiunta ai campioni pre-step. Verificati numero e ordine dei campioni, fasi, coordinate goal, distanza finale e corrispondenza con la diagnostica post-step. Nessuna simulazione avviata.','',
 'Riproduzione offline:', '```bash',
 'PYTHONNOUSERSITE=1 /usr/bin/python3 spot_rl_energy_navigation/train_general_big_map_test6_SAC/analysis/paths_stage1_300k_20261029/plot_paths.py','```'
]
(OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
print('\n'.join(table))
print('Minimum sampled distances:',minimum)
print('Output:',OUT)
