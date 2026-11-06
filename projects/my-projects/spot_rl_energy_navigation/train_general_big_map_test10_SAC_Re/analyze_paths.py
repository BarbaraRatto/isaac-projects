"""Offline path/terrain/diagnostic report for play.py results. No simulator needed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Circle
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

TERRAINS = {'t1_asphalt':('#d1d4cf','Grigio'), 'lab_green':('#9ecc59','Verde resistente')}
REASONS = {'goal':'Successo', 'fallen':'Caduta', 'timeout':'Timeout',
           'out_of_bounds':'Fuori mappa', 'alignment_timeout':'Timeout allineamento'}
COLORS = {'goal':'#147d43','fallen':'#c92e4c','timeout':'#b97800','out_of_bounds':'#7d3c98','alignment_timeout':'#777777'}
BLUE, ORANGE = '#155da2', '#ed850e'


def csv_rows(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f'Truncated or malformed CSV: {path}')
    return rows


def terrain_at(meta,x,y):
    col=int(np.floor((x-meta['bounds_min'][0])/meta['cell_width_m']))
    row=int(np.floor((y-meta['bounds_min'][1])/meta['cell_height_m']))
    if row<0 or col<0 or row>=len(meta['grid']) or col>=len(meta['grid'][0]): return 'outside'
    return meta['grid'][row][col]


def label(name):
    return TERRAINS.get(name,('',name))[1]


def background(ax,meta):
    x0,y0=meta['bounds_min'][:2];w,h=meta['cell_width_m'],meta['cell_height_m']
    for r,row in enumerate(meta['grid']):
        for c,name in enumerate(row):
            ax.add_patch(Rectangle((x0+c*w,y0+r*h),w,h,facecolor=TERRAINS[name][0],edgecolor='white',lw=.7,zorder=0))
    ax.set_aspect('equal');ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]')
    ax.grid(alpha=.15)


def path_plot(ax,meta,m,zoom=True):
    e,xy,diag=m
    background(ax,meta)
    start=np.array([e['start_x'],e['start_y']]);goal=np.array([e['goal_x'],e['goal_y']])
    n=e['alignment_steps']
    ax.plot([start[0],goal[0]],[start[1],goal[1]],':',color='#555555',lw=1)
    ax.plot(xy[:n+1,0],xy[:n+1,1],color=ORANGE,lw=2.2)
    ax.plot(xy[n:,0],xy[n:,1],color=BLUE,lw=1.7)
    for j in np.linspace(n,max(n,len(xy)-3),4,dtype=int):
        k=min(j+2,len(xy)-1)
        ax.annotate('',xy=xy[k],xytext=xy[j],arrowprops=dict(arrowstyle='->',color=BLUE,lw=1))
    ax.scatter(*start,color='black',s=28,zorder=5)
    ax.scatter(*goal,color='#147d43',marker='*',s=120,zorder=5)
    ax.add_patch(Circle(goal,.55,fill=False,color='#147d43',lw=1))
    ax.scatter(*xy[-1],color=COLORS[e['reason']],marker='x',s=65,zorder=6)
    ax.annotate('A',start,xytext=(5,5),textcoords='offset points',weight='bold')
    ax.annotate('B',goal,xytext=(5,5),textcoords='offset points',weight='bold')
    if zoom:
        p=np.vstack([xy,start,goal]);lo=p.min(0)-.8;hi=p.max(0)+.8
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
    ax.set_title(f"E{e['episode']} · {REASONS[e['reason']]}\n{e['path_m']:.2f} m percorsi · {e['energy_j']:.0f} J",fontsize=11)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--policy-label',help='label shown on all exported figures; default derived from checkpoint')
    parser.add_argument('--map-metadata',type=Path,default=Path(__file__).resolve().parent/'artifacts/map_metadata.json')
    args=parser.parse_args()
    out=args.output or args.results/'analysis';out.mkdir(parents=True,exist_ok=True)
    doc=json.loads((args.results/'summary.json').read_text())
    policy_label=args.policy_label or Path(doc['checkpoint']).stem.replace('policy_','Policy ').replace('_',' ')
    meta=json.loads(args.map_metadata.read_text())
    trajectories=csv_rows(args.results/'trajectories.csv')
    diagnostics=csv_rows(args.results/'fall_diagnostics.csv')
    episode_rows=csv_rows(args.results/'episodes.csv')
    missions=[];metrics=[];all_steps=[]
    for e in doc['episodes']:
        match=lambda r:int(r['env_id'])==e['env_id'] and int(r['episode'])==e['episode']
        rows=sorted(filter(match,trajectories),key=lambda r:int(r['step']))
        diag=sorted(filter(match,diagnostics),key=lambda r:int(r['step']))
        er=list(filter(match,episode_rows));assert len(er)==1
        assert er[0]['reason']==e['reason']
        np.testing.assert_allclose(float(er[0]['energy_j']),e['energy_j'])
        assert [int(r['step']) for r in rows]==list(range(e['steps']))
        assert [int(r['step']) for r in diag]==list(range(1,e['steps']+1))
        assert sum(r['phase']=='alignment' for r in rows)==e['alignment_steps']
        assert int(e['alignment_steps'])+int(e['navigation_steps'])==e['steps']
        np.testing.assert_allclose([[float(r['goal_x']),float(r['goal_y'])] for r in rows],
                                   np.tile([e['goal_x'],e['goal_y']],(len(rows),1)),atol=1e-4)
        xy=np.array([[float(r['x']),float(r['y'])] for r in rows]+[[e['final_x'],e['final_y']]])
        np.testing.assert_allclose(xy[1:],[[float(r['x']),float(r['y'])] for r in diag],atol=1e-4)
        np.testing.assert_allclose(np.linalg.norm(xy[-1]-[e['goal_x'],e['goal_y']]),e['final_distance_m'],atol=1e-4)
        np.testing.assert_allclose(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum(),e['path_m'],atol=2e-4)
        # Check terrain orientation against the recorded start/goal labels.
        assert terrain_at(meta,e['start_x'],e['start_y'])==e['start_terrain']
        assert terrain_at(meta,e['goal_x'],e['goal_y'])==e['goal_terrain']
        dt=float(diag[-1]['time_s'])/e['steps'];all_steps.append(dt)
        first_tilt=next((r for r in diag if float(r['tilt_deg'])>=90),None)
        tail=xy[max(0,len(xy)-1-round(10/dt)):]
        tail_dist=float(np.linalg.norm(np.diff(tail,axis=0),axis=1).sum())
        nav=[r for r in rows if r['phase']=='navigation']
        mean_command=float(np.mean([float(r['command_vx']) for r in nav])) if nav else 0.
        last_nav=nav[-round(10/dt):]
        last_command=float(np.mean([float(r['command_vx']) for r in last_nav])) if last_nav else 0.
        d=dict(episode=e['episode'],env_id=e['env_id'],outcome=e['reason'],
               initial_distance_m=e['initial_distance_m'],path_m=e['path_m'],
               final_distance_m=e['final_distance_m'],duration_s=e['steps']*dt,
               energy_j=e['energy_j'],energy_per_path_m=e['energy_j']/max(e['path_m'],1e-9),
               path_over_initial_distance=e['path_m']/e['initial_distance_m'],
               final_terrain=terrain_at(meta,e['final_x'],e['final_y']),fall_cause=e['fall_cause'],
               max_tilt_deg=e['max_tilt_deg'],last_10s_path_m=tail_dist,
               mean_navigation_command_vx=mean_command,last_10s_command_vx=last_command,
               first_sample_tilt90_time_s=float(first_tilt['time_s']) if first_tilt else None,
               first_sample_tilt90_terrain=terrain_at(meta,float(first_tilt['x']),float(first_tilt['y'])) if first_tilt else None)
        metrics.append(d);missions.append((e,xy,diag))
    assert sum(e['steps'] for e,_,_ in missions)==len(trajectories)==len(diagnostics)
    assert len(missions)==doc['summary']['episodes']
    assert sum(e['success'] for e,_,_ in missions)==doc['summary']['successes']
    np.testing.assert_allclose(all_steps,all_steps[0]);dt=all_steps[0]
    transitions=sorted({int(r['transitions']) for r in episode_rows})
    scenario=f"suite {len(doc['mission_suite']['cases'])} missioni" if doc.get('mission_suite') else f"fase {doc['summary']['stage']}"
    title=f"Test10 SAC Re · {policy_label} · {scenario}\nCheckpoint {','.join(map(str,transitions))} transizioni"
    terrain_legend=[Patch(facecolor=color,label=name) for color,name in TERRAINS.values()]
    path_legend=[Line2D([],[],color=ORANGE,label='Allineamento'),Line2D([],[],color=BLUE,label='Navigazione'),
                 Line2D([],[],marker='o',color='black',ls='',label='A: partenza'),
                 Line2D([],[],marker='*',color='#147d43',ls='',label='B: obiettivo (raggio 0,55 m)'),
                 Line2D([],[],marker='x',color='#c92e4c',ls='',label='Posizione finale')]
    fig,ax=plt.subplots(figsize=(16,9));background(ax,meta)
    for e,xy,_ in missions:
        color=COLORS[e['reason']]
        ax.plot(xy[:,0],xy[:,1],color=color,lw=2,alpha=.85)
        ax.scatter(*xy[0],color=color,edgecolor='black',s=32,zorder=5)
        ax.scatter(e['goal_x'],e['goal_y'],color=color,marker='*',s=110,zorder=5)
        ax.scatter(*xy[-1],color=color,marker='x',s=65,zorder=6)
        ax.annotate(f"E{e['episode']}",xy[0],xytext=(5,6),textcoords='offset points',weight='bold',fontsize=10,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
    ax.set_xlim(meta['bounds_min'][0]-.5,meta['bounds_max'][0]+.5)
    ax.set_ylim(meta['bounds_min'][1]-.5,meta['bounds_max'][1]+.5)
    ax.set_title(title+'\n'+f"{doc['summary']['successes']}/{len(missions)} successi · pallino: partenza · stella: obiettivo · croce: fine",fontsize=15,pad=16)
    fig.legend(handles=[Line2D([],[],color=COLORS[k],lw=3,label=v) for k,v in REASONS.items() if any(e['reason']==k for e,_,_ in missions)],loc='lower center',bbox_to_anchor=(.5,.075),ncol=3)
    fig.legend(handles=terrain_legend,loc='lower center',bbox_to_anchor=(.5,.035),ncol=2)
    fig.text(.5,.015,'Sfondo: celle dei terreni della mappa congelata; verde: resistenza viscosa 90 N·s/m.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.14,1,.93));fig.savefig(out/'mappa_percorsi.png',dpi=170,bbox_inches='tight',pad_inches=.2);fig.savefig(out/'mappa_percorsi.pdf',bbox_inches='tight',pad_inches=.2);plt.close(fig)
    fig,axes=plt.subplots((len(missions)+1)//2,2,figsize=(16,4.8*((len(missions)+1)//2)),squeeze=False)
    for ax,m in zip(axes.flat,missions):path_plot(ax,meta,m)
    for ax in list(axes.flat)[len(missions):]:ax.axis('off')
    fig.suptitle(title+'\nPercorsi separati — scale diverse tra episodi',fontsize=17)
    fig.legend(handles=path_legend,loc='lower center',bbox_to_anchor=(.5,.025),ncol=3)
    fig.legend(handles=terrain_legend,loc='lower center',bbox_to_anchor=(.5,.008),ncol=2)
    fig.tight_layout(rect=(0,.065,1,.96),h_pad=2.5)
    fig.savefig(out/'percorsi_per_episodio.png',dpi=140);fig.savefig(out/'percorsi_per_episodio.pdf');plt.close(fig)
    with PdfPages(out/'diagnostica_episodi.pdf') as pdf:
        for m,d in zip(missions,metrics):
            e,xy,diag=m
            fig=plt.figure(figsize=(15,8));gs=fig.add_gridspec(2,2,width_ratios=[1.2,1])
            ax=fig.add_subplot(gs[:,0]);path_plot(ax,meta,m)
            distax=fig.add_subplot(gs[0,1]);tiltax=fig.add_subplot(gs[1,1])
            times=np.arange(len(xy))*dt
            distax.plot(times,np.linalg.norm(xy-[e['goal_x'],e['goal_y']],axis=1),color=BLUE)
            distax.axhline(.55,color='#147d43',ls='--',label='Raggio obiettivo')
            distax.axvspan(0,e['alignment_steps']*dt,color=ORANGE,alpha=.15,label='Allineamento')
            distax.set(xlabel='Tempo [s]',ylabel='Distanza dal goal [m]');distax.legend();distax.grid(alpha=.2)
            tiltax.plot([float(r['time_s']) for r in diag],[float(r['tilt_deg']) for r in diag],color='#c92e4c',label='Inclinazione campionata')
            tiltax.axhline(90,color='black',ls='--',label='90° (caduta solo se persistente)')
            tiltax.set(xlabel='Tempo [s]',ylabel='Inclinazione [°]',ylim=(0,190));tiltax.legend(fontsize=9);tiltax.grid(alpha=.2)
            fig.suptitle(f"{policy_label} · E{e['episode']} · {REASONS[e['reason']]} · {e['steps']*dt:.1f} s · {e['energy_j']:.0f} J\n"
                         f"Fine nella cella: {label(d['final_terrain'])} · causa registrata: {e['fall_cause'] or 'nessuna'}",fontsize=14)
            fig.legend(handles=path_legend,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,fontsize=9)
            fig.legend(handles=terrain_legend,loc='lower center',bbox_to_anchor=(.5,.007),ncol=2,fontsize=9)
            fig.tight_layout(rect=(0,.14,1,.91),h_pad=2);pdf.savefig(fig);plt.close(fig)
    with (out/'riepilogo_episodi.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    (out/'analisi.json').write_text(json.dumps(dict(source=str(args.results.resolve()),checkpoint=doc['checkpoint'],
        checkpoint_transitions=transitions,sample_dt_s=dt,summary=doc['summary'],episodes=metrics),indent=2)+'\n')
    report=[f'# {title}','',f"Checkpoint: `{doc['checkpoint']}`",'',
      f"Esiti: **{doc['summary']['successes']}/{len(missions)} successi**. {doc['summary']['outcomes']}",
      f"Allineamento riuscito: {100*doc['summary']['alignment_success_rate']:.0f}%. Energia media sui soli successi: {doc['summary']['mean_success_energy_j']:.1f} J.",
      '', '| Episodio | Esito | Percorso (m) | Tempo (s) | Energia (J) | Distanza finale (m) | Cella finale |',
      '|---|---|---:|---:|---:|---:|---|']
    for d in metrics:
        report.append(f"| {d['episode']} | {REASONS[d['outcome']]} | {d['path_m']:.2f} | {d['duration_s']:.1f} | {d['energy_j']:.0f} | {d['final_distance_m']:.2f} | {label(d['final_terrain'])} |")
    report += ['', '## Lettura e limiti', '',
      '- A = partenza; B = obiettivo; croce = posizione finale. Arancione = allineamento; blu = navigazione nei pannelli separati.',
      '- Lo sfondo mostra le celle dei materiali, con verde resistente e grigio senza carico. La cella del punto finale non dimostra la causa della caduta.',
      '- Il tratto finale è ricostruito da summary.json: trajectories.csv contiene campioni prima del passo; fall_diagnostics.csv dopo il passo, prima del reset.',
      '- Verificati: conteggio e ordine dei campioni, corrispondenza pre/post-step, fasi, obiettivi, distanza finale, lunghezza del percorso e terreni di partenza/arrivo.',
      '- Le curve di inclinazione sono campionate ogni 0,2 s; i massimi dei record includono i sottopassi fisici. Un picco isolato non equivale alla condizione di caduta persistente.',
      '- Le durate sono tempo simulato, non tempo reale di esecuzione. Energia e percorso includono l’allineamento.',
      '- Il rapporto percorso/distanza iniziale è solo un riferimento geometrico: la retta può attraversare ostacoli e il goal ha raggio 0,55 m.',
      '- Una singola suite e un solo seed non bastano per concludere che una policy sia migliore di un’altra. Confrontare gli stessi start, goal e orientamenti; aumentare il numero di episodi.',
      '- Le missioni non hanno una verifica preventiva di raggiungibilità; un fallimento non prova da solo un difetto della policy.',
      '', '## Osservazioni sui fallimenti','']
    for d in metrics:
        if d['outcome']=='fallen':
            report.append(f"- E{d['episode']}: causa registrata `{d['fall_cause']}`, massimo tilt {d['max_tilt_deg']:.1f}°; prima inclinazione campionata ≥90° a {d['first_sample_tilt90_time_s']} s nella cella {label(d['first_sample_tilt90_terrain'])}; fine in {label(d['final_terrain'])}.")
        elif d['outcome']=='timeout':
            report.append(f"- E{d['episode']}: timeout, distanza residua {d['final_distance_m']:.2f} m, ultimi 10 s percorsi {d['last_10s_path_m']:.2f} m, comando avanti medio negli ultimi 10 s {d['last_10s_command_vx']:.2f} m/s, massimo tilt {d['max_tilt_deg']:.1f}°. Compatibile con difficoltà di avanzamento; i dati non distinguono da soli incastro/contatto/postura.")
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(checkpoint_transitions=transitions,summary=doc['summary'],episodes=metrics),indent=2))
    print('Outputs:',out)


if __name__=='__main__':main()
