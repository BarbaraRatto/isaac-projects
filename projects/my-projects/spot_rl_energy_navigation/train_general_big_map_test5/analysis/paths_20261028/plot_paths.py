"""Offline plots of the completed test5 evaluation; never launches Isaac Sim."""
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
SOURCE=ROOT/'results/evaluation_20261028_114806_451932'
doc=json.loads((SOURCE/'summary.json').read_text())
with (SOURCE/'episodes.csv').open() as stream: episode_rows=list(csv.DictReader(stream))
transitions={int(row['transitions']) for row in episode_rows}
assert len(transitions)==1
transitions=transitions.pop()
assert transitions==200000
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
fig.suptitle(f"Test 5 — policy a {transitions:,} transizioni — fase {stage}\n{summary['episodes']} episodi: {summary['successes']} goal raggiunti, {summary['outcomes']['fallen']} cadute, {summary['outcomes']['timeout']} timeout | Allineamento riuscito: {sum(e['alignment_success'] for e,_ in missions)}/9",fontsize=17,y=.985)
fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.052),fontsize=10)
fig.legend(handles=terrain_handles,loc='lower center',ncol=7,bbox_to_anchor=(.5,.027),fontsize=9)
fig.text(.5,.012,'Coordinate della mappa in metri; scale diverse per episodio. Percorso ed energia totali includono l’allineamento.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.105,1,.945),h_pad=4,w_pad=2.5)
fig.savefig(OUT/'percorsi_test5_200k.pdf')
fig.savefig(OUT/'percorsi_test5_200k.png',dpi=150)
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
  fig.suptitle(f"Test 5 — episodio {e['episode']} — allineamento {e['alignment_steps']*DT:.1f} s / {e['alignment_energy_j']:.0f} J; navigazione {e['navigation_steps']*DT:.1f} s / {e['navigation_energy_j']:.0f} J",fontsize=13)
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

# Post-step diagnostic samples include the exact terminal state before reset.
with (SOURCE/'fall_diagnostics.csv').open() as stream:
 diagnostic_rows=list(csv.DictReader(stream))
with PdfPages(OUT/'diagnostica_cadute.pdf') as pdf:
 for e,xy in missions:
  ds=sorted([r for r in diagnostic_rows if int(r['env_id'])==e['env_id'] and int(r['episode'])==e['episode']],key=lambda r:int(r['step']))
  assert [int(r['step']) for r in ds]==list(range(1,e['steps']+1))
  times=np.array([float(r['time_s']) for r in ds])
  np.testing.assert_allclose(times,np.arange(1,e['steps']+1)*DT,atol=1e-5)
  np.testing.assert_allclose([[float(r['x']),float(r['y'])] for r in ds],xy[1:],atol=1e-4)
  assert ds[-1]['fall_cause']==e['fall_cause']
  if e['reason']!='fallen':continue
  fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
  axes[0].plot(times,[float(r['tilt_deg']) for r in ds],color=BLUE,label='Inclinazione campionata')
  axes[0].axhline(nav['fall_tilt_deg'],color=RED,ls='--',label='Soglia di inclinazione')
  axes[0].set(ylabel='Inclinazione [gradi]',ylim=(0,180))
  axes[1].plot(times,[float(r['tilt_duration_s']) for r in ds],label='Inclinazione oltre soglia',color=BLUE)
  axes[1].plot(times,[float(r['body_contact_duration_s']) for r in ds],label='Contatto del corpo col terreno',color=ORANGE)
  axes[1].axhline(nav['fall_persistence_seconds'],ls='--',color=RED,label='Durata richiesta per terminare')
  axes[1].set(ylabel='Durata consecutiva [s]')
  axes[2].plot(times,[float(r['base_clearance_m']) if r['base_clearance_m'] else np.nan for r in ds],color=BLUE)
  axes[2].axhline(nav['minimum_body_height_m'],ls=':',color='gray',label='Vecchia soglia: solo diagnostica')
  axes[2].set(ylabel='Distanza base–terreno [m]',xlabel='Tempo dalla partenza [s]')
  for ax in axes:
   ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
  fig.suptitle(f"Test 5 — episodio {e['episode']}: caduta per inclinazione persistente\nInclinazione massima {e['max_tilt_deg']:.1f}°; permanenza oltre soglia {e['max_tilt_duration_s']:.2f} s",fontsize=13)
  fig.text(.5,.015,'Log ogni 0,2 s; contatori di persistenza aggiornati ogni 0,005 s. Altezza mostrata solo a scopo diagnostico.',ha='center',fontsize=9)
  fig.tight_layout(rect=(0,.035,1,.93))
  pdf.savefig(fig)
  fig.savefig(OUT/f"caduta_episodio_{e['episode']}.png",dpi=130)
  plt.close(fig)

farther=[e['episode'] for e,_ in missions if e['final_distance_m']>e['initial_distance_m']]
report=[
 '# Test 5 — risultati a 200.000 transizioni', '',
 f"Fonte: `{SOURCE}`", f"Checkpoint: `{doc['checkpoint']}`", '',
 f"Fase {stage}: {summary['stage_name']}. Seed {doc['seed']}; {doc['num_envs']} robot. "
 f"Successi {summary['successes']}/{summary['episodes']}; {summary['outcomes']['timeout']} timeout e {summary['outcomes']['fallen']} caduta. Allineamento riuscito in tutti i 9 episodi.",
 '',*table,'',
 'Tempo, percorso ed energia sono totali, incluso l’allineamento iniziale. A–B diretto è la distanza euclidea, non un percorso ottimo sul terreno. Il goal è raggiunto entro 0,55 m.', '',
 '## Lettura dei risultati','',
 f"Gli episodi {', '.join(map(str,farther))} terminano più lontano dal goal rispetto alla partenza. Negli otto timeout Spot percorre da 7,34 a 20,37 m senza raggiungere goal inizialmente distanti 3,47–5,99 m. Queste missioni indicano un problema di navigazione; il timeout è l’esito, ma non dimostra che serva soltanto più tempo.", '',
 'Nell’episodio 8 il criterio di inclinazione conferma il ribaltamento: massimo 165,12°, permanenza oltre 90° di 1,04 s. Il contatto continuo massimo del corpo dura 0,205 s; la causa registrata è quindi tilt, non body_contact. Negli altri episodi l’inclinazione massima resta sotto 14°.', '',
 'Nell’episodio 6 è registrato un campione con misura dell’altezza non valida: il vecchio criterio avrebbe richiesto la terminazione in quel punto, mentre il nuovo ha consentito di continuare. Questo non permette di ricostruire cosa sarebbe successo con una policy allenata diversamente.', '',
 'Questo test di nove missioni ha risultati insufficienti, ma da solo non identifica la causa né isola l’effetto delle singole modifiche al training.', '',
 '## Grafici','',
 '- `percorsi_test5_200k.pdf` e `.png`: nove percorsi con start A nero, goal B verde e raggio di successo, allineamento arancione, navigazione blu, punto finale a croce rossa. Freccia nera: orientamento iniziale; rombo arancione: inizio RL. Scale diverse per episodio.',
 '- `episodi_dettaglio.pdf`: nove pagine con percorso e distanza dal goal nel tempo.',
 '- `diagnostica_cadute.pdf` e `caduta_episodio_8.png`: inclinazione, durata delle condizioni di caduta e altezza della base.',
 '- `riepilogo_episodi.csv`: tabella con coordinate, terreni, energia e tempi separati, diagnostica terminale.', '',
 'I terreni sono quelli della mappa congelata del test 5. Il segmento tratteggiato A–B è solo un riferimento geometrico. La posa terminale esatta è aggiunta al CSV delle traiettorie, che registra campioni prima del passo. Verificati conteggi, ordine, fasi, goal e distanze finali; i campioni diagnostici dopo ogni passo coincidono con le pose successive delle traiettorie. Nessuna simulazione avviata.', '',
 'Riproduzione offline:', '', '```bash',
 'PYTHONNOUSERSITE=1 /usr/bin/python3 spot_rl_energy_navigation/train_general_big_map_test5/analysis/paths_20261028/plot_paths.py', '```',
]
(OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
print('\n'.join(table))
print('Generated plots for',len(missions),'episodes in',OUT)
