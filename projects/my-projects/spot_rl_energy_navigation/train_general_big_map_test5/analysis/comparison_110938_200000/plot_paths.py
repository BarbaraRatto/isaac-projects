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
SOURCE=ROOT/'results/evaluation_20261028_124938_174249'
doc=json.loads((SOURCE/'summary.json').read_text())
with (SOURCE/'episodes.csv').open() as stream: episode_rows=list(csv.DictReader(stream))
transitions={int(row['transitions']) for row in episode_rows}
assert len(transitions)==1
transitions=transitions.pop()
assert transitions==110938
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
fig.suptitle(f"Test 5 — policy a {transitions:,} transizioni — fase {stage}\n{summary['episodes']} episodi: {summary['successes']} goal raggiunti, {summary['outcomes']['fallen']} cadute, {summary['outcomes']['timeout']} timeout, {summary['outcomes']['out_of_bounds']} fuori mappa | Allineamento riuscito: {sum(e['alignment_success'] for e,_ in missions)}/9",fontsize=17,y=.985)
fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.052),fontsize=10)
fig.legend(handles=terrain_handles,loc='lower center',ncol=7,bbox_to_anchor=(.5,.027),fontsize=9)
fig.text(.5,.012,'Coordinate della mappa in metri; scale diverse per episodio. Percorso ed energia totali includono l’allineamento.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.105,1,.945),h_pad=4,w_pad=2.5)
fig.savefig(OUT/'percorsi_test5_110938.pdf')
fig.savefig(OUT/'percorsi_test5_110938.png',dpi=150)
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

FINAL_SOURCE=ROOT/'results/evaluation_20261028_114806_451932'
final_doc=json.loads((FINAL_SOURCE/'summary.json').read_text())
assert doc['seed']==final_doc['seed']==20000
assert doc['num_envs']==final_doc['num_envs']==1
assert stage==final_doc['summary']['stage']==1
with (FINAL_SOURCE/'trajectories.csv').open() as stream:
 final_rows=list(csv.DictReader(stream))
final_missions=[]
for e,xy in missions:
 other=next(r for r in final_doc['episodes'] if r['episode']==e['episode'])
 for key in ('start_id','goal_id','start_x','start_y','goal_x','goal_y','initial_yaw','band','deadline_s'):
  assert e[key]==other[key],(e['episode'],key)
 r=sorted([r for r in final_rows if int(r['episode'])==e['episode'] and int(r['env_id'])==e['env_id']],key=lambda r:int(r['step']))
 assert [int(r['step']) for r in r]==list(range(other['steps']))
 other_xy=np.array([[float(r['x']),float(r['y'])] for r in r]+[[other['final_x'],other['final_y']]])
 np.testing.assert_allclose(np.linalg.norm(other_xy[-1]-[other['goal_x'],other['goal_y']]),other['final_distance_m'],atol=1e-4)
 np.testing.assert_allclose(xy[:e['alignment_steps']+1],other_xy[:other['alignment_steps']+1],atol=1e-4)
 final_missions.append((other,other_xy))

PURPLE='#942b91'
fig,axes=plt.subplots(3,3,figsize=(18,18))
for ax,mission,other_mission in zip(axes.flat,missions,final_missions):
 e,xy=mission;other,other_xy=other_mission
 draw(ax,mission)
 ax.plot(other_xy[:,0],other_xy[:,1],color=PURPLE,lw=1.5,ls='--',zorder=3)
 ax.scatter(*other_xy[-1],color=PURPLE,marker='X',s=55,zorder=8)
 points=np.vstack((xy,other_xy,[e['start_x'],e['start_y']],[e['goal_x'],e['goal_y']]))
 lo=points.min(0)-.9;hi=points.max(0)+.9
 ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
 ax.set_title(f"Episodio {e['episode']}\n110.938: {REASONS[e['reason']]} | 200.000: {REASONS[other['reason']]}",fontsize=10,pad=12)
comparison_handles=[handles[0],handles[1],handles[2],
 Line2D([],[],color=BLUE,lw=2,label='110.938: navigazione RL'),
 Line2D([],[],color=PURPLE,ls='--',lw=2,label='200.000: percorso'),
 Line2D([],[],color=RED,marker='x',ls='',label='Fine episodio a 110.938'),
 Line2D([],[],color=PURPLE,marker='X',ls='',label='Fine episodio a 200.000')]
fig.suptitle('Test 5 — confronto sulle stesse nove missioni\n110.938 transizioni: 3/9 successi | 200.000 transizioni: 0/9 successi',fontsize=17,y=.985)
fig.legend(handles=comparison_handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.048),fontsize=10)
fig.legend(handles=terrain_handles,loc='lower center',ncol=7,bbox_to_anchor=(.5,.022),fontsize=9)
fig.text(.5,.008,'Coordinate della mappa; scale diverse tra episodi, uguali per i due checkpoint in ciascun riquadro.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.12,1,.945),h_pad=4,w_pad=2.5)
fig.savefig(OUT/'confronto_percorsi.pdf');fig.savefig(OUT/'confronto_percorsi.png',dpi=130);plt.close(fig)

fig,axes=plt.subplots(3,3,figsize=(15,11))
for ax,(e,xy),(other,other_xy) in zip(axes.flat,missions,final_missions):
 for coords,color,label in ((xy,BLUE,'110.938'),(other_xy,PURPLE,'200.000')):
  ax.plot(np.arange(len(coords))*DT,np.linalg.norm(coords-[e['goal_x'],e['goal_y']],axis=1),color=color,label=label)
 ax.axhline(RADIUS,ls=':',color=GREEN,label='Raggio goal')
 ax.axvspan(0,e['alignment_steps']*DT,color=ORANGE,alpha=.12)
 ax.set(title=f"Episodio {e['episode']}",xlabel='Tempo [s]',ylabel='Distanza dal goal [m]',ylim=(0,None))
 ax.grid(alpha=.2)
fig.suptitle('Distanza dal goal nel tempo — confronto dei checkpoint',fontsize=16)
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3)
fig.tight_layout(rect=(0,.04,1,.95))
fig.savefig(OUT/'confronto_distanze.pdf');plt.close(fig)

comparison_table=['| Episodio | Esito a 110.938 | Esito a 200.000 | Distanza finale 110.938 (m) | Distanza finale 200.000 (m) |',
 '|---|---|---|---:|---:|']
comparison_records=[]
for (e,xy),(other,other_xy) in zip(missions,final_missions):
 comparison_table.append(f"| {e['episode']} | {REASONS[e['reason']]} | {REASONS[other['reason']]} | {e['final_distance_m']:.2f} | {other['final_distance_m']:.2f} |")
 row={'episode':e['episode']}
 for suffix,item in (('110938',e),('200000',other)):
  for key in ('reason','path_m','energy_j','final_distance_m'):
   row[key+'_'+suffix]=item[key]
 comparison_records.append(row)
with (OUT/'confronto_episodi.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(comparison_records[0]));writer.writeheader();writer.writerows(comparison_records)

fall_notes=[]
for e,xy in missions:
 if e['reason']=='fallen':
  fall_notes.append(f"- Episodio {e['episode']}: causa `{e['fall_cause']}`, inclinazione massima {e['max_tilt_deg']:.2f}°, permanenza massima oltre 90° {e['max_tilt_duration_s']:.2f} s. Coordinate terminali ({e['final_x']:.2f}, {e['final_y']:.2f}) m. Le coordinate della base non identificano da sole il terreno di contatto delle zampe.")
report=[
 '# Test 5 — confronto 110.938 / 200.000 transizioni','',
 f"Nuova valutazione: `{SOURCE}`",f"Checkpoint: `{doc['checkpoint']}`",f"Valutazione finale di confronto: `{FINAL_SOURCE}`",'',
 'Stessa mappa, fase 1, seed 20000 e un robot. Verificata uguaglianza di start, goal, orientamenti iniziali, deadline e traiettorie di allineamento. Nessuna simulazione o modifica del training effettuata.','',
 '## Risultati del checkpoint a 110.938','',*table,'',
 'Totale: 3/9 successi (33,3%), 3 timeout, 2 cadute e 1 uscita dalla mappa. Allineamento riuscito 9/9. Tempi, percorso ed energia includono l’allineamento automatico. La distanza A–B è euclidea, non un percorso ottimo calcolato sul terreno. Raggio goal: 0,55 m.','',
 f"Nei tre successi: percorso medio {summary['mean_success_path_m']:.2f} m, energia media {summary['mean_success_energy_j']:.0f} J, durata media {summary['mean_success_seconds']:.2f} s.",'',
 '## Confronto episodio per episodio','',*comparison_table,'',
 'Le missioni 5, 8 e 9 riescono a 110.938 transizioni e falliscono a 200.000. La missione 5 raggiunge il goal sulle rocce con un percorso di 5,82 m, contro 5,99 m di distanza iniziale: una lunghezza inferiore è possibile perché basta entrare nel raggio del goal. L’episodio 8 raggiunge un goal su rampa in 9 s; il modello finale vi si ribalta. L’episodio 9 raggiunge il goal in 11,4 s, mentre quello finale resta vicino allo start compiendo giri.','',
 'La policy intermedia resta inaffidabile: negli episodi 2 e 4 termina più lontano dal goal dopo circa 24 m; nell’episodio 7 percorre 22,36 m senza arrivare. Nell’episodio 3 esce dalla mappa. Non aveva già risolto la navigazione generale.','',
 '## Cadute','',*fall_notes,'',
 '## Conclusione e limiti','',
 'Su queste nove missioni il checkpoint intermedio è migliore per raggiungimento del goal, ma ha più cadute e uscite dalla mappa. Il peggioramento finale non si spiega soltanto con un numero insufficiente di transizioni: durante questo allenamento sono andati persi alcuni comportamenti utili. Anche le valutazioni interne della fase 1 erano passate da 12/32 successi a 110.938 a 7/32 a 200.000.','',
 'Queste osservazioni non identificano la causa del peggioramento: non dimostrano che dipenda dalla penalità di orientamento, dalla capacità della rete o dalle soglie di caduta. Una sola run e nove missioni non stimano con precisione la prestazione generale. L’energia di episodi falliti e riusciti non è un confronto di efficienza energetica a parità di task completata.','',
 'Conservare il checkpoint intermedio come riferimento. Prima di prolungare o modificare ancora i pesi, analizzare gli aggiornamenti PPO e la distribuzione delle missioni nel tratto 110.938–200.000; una successiva prova controllata dovrebbe modificare un solo fattore.','',
 '## File','',
 '- `percorsi_test5_110938.pdf` / `.png`: start A nero, goal B verde, allineamento arancione, navigazione blu, posizione finale rossa.',
 '- `episodi_dettaglio.pdf`: percorso e distanza dal goal per ogni episodio.',
 '- `confronto_percorsi.pdf` / `.png`: traiettorie sovrapposte; blu 110.938, viola tratteggiato 200.000.',
 '- `confronto_distanze.pdf`: distanza dal goal nel tempo per entrambi i checkpoint.',
 '- `diagnostica_cadute.pdf`: inclinazione e persistenza nelle due cadute.',
 '- `riepilogo_episodi.csv` e `confronto_episodi.csv`: dati numerici.','',
 'Verificati numero e ordine dei campioni, fasi, goal, pose terminali e corrispondenza dei log diagnostici post-step.','',
 'Riproduzione:', '```bash',
 'PYTHONNOUSERSITE=1 /usr/bin/python3 spot_rl_energy_navigation/train_general_big_map_test5/analysis/comparison_110938_200000/plot_paths.py','```',
]
(OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
print('\n'.join(table));print('\n'.join(fall_notes));print('Output:',OUT)

