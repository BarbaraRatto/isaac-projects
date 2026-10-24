"""Plot recorded evaluation speeds without starting Isaac Sim or loading a policy."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluations', type=Path, nargs=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    stats = []
    samples = []
    for i, directory in enumerate(args.evaluations):
        summary = json.loads((directory/'summary.json').read_text())
        checkpoint = Path(summary['checkpoint'])
        config = json.loads((checkpoint.parent/'config.json').read_text())
        nav = config['signature']['navigation']
        dt = nav['physics_dt']*nav['physics_steps_per_action']
        vmax = nav['max_forward_m_s']
        rows = list(csv.DictReader((directory/'trajectories.csv').open()))
        episodes = {(int(r['env_id']), int(r['episode'])):r for r in summary['episodes']}
        commands, actual = [], []
        matrices = [[], []]
        title = 'Asfalto a sinistra (rocce a destra)' if summary['layout'].endswith('right_rocks') else 'Asfalto a destra (rocce a sinistra)'
        for key, record in sorted(episodes.items()):
            r = sorted([r for r in rows if (int(r['env_id']),int(r['episode']))==key], key=lambda r:int(r['step']))
            t = np.array([int(x['step'])*dt for x in r])
            assert len(r)==record['steps'] and np.allclose(np.diff(t),dt)
            xy = np.array([[float(x['x']),float(x['y'])] for x in r])
            action = np.array([float(x['action_forward']) for x in r])
            cmd = (np.clip(action,-1,1)+1)*vmax/2
            measured = np.linalg.norm(np.diff(xy,axis=0),axis=1)/np.diff(t)
            commands.extend(cmd.tolist()); actual.extend(measured.tolist())
            matrices[0].append(cmd); matrices[1].append(measured)
            axes[i,0].step(t,cmd,where='post',color='#1976b8',alpha=.22,lw=1)
            axes[i,1].plot(t[:-1]+dt/2,measured,color='#d26a16',alpha=.22,lw=1)
            for j, time in enumerate(t):
                samples.append(dict(layout=summary['layout'],env_id=key[0],episode=key[1],
                    time_s=time,command_vx_m_s=cmd[j],
                    measured_horizontal_m_s=measured[j] if j<len(measured) else ''))
        for column, arrays in enumerate(matrices):
            matrix = np.full((len(arrays), max(map(len,arrays))),np.nan)
            for k,a in enumerate(arrays): matrix[k,:len(a)]=a
            support = np.isfinite(matrix).sum(0)
            mean = np.nanmean(matrix,axis=0)
            mean[support<3]=np.nan
            time = np.arange(len(mean))*dt+(dt/2 if column else 0.)
            ax=axes[i,column]
            if column==0: ax.step(time,mean,where='post',color='#1976b8',lw=2.5,label='Media (almeno 3 episodi attivi)')
            else: ax.plot(time,mean,color='#d26a16',lw=2.5,label='Media (almeno 3 episodi attivi)')
            ax.axhline(.9,color='#444444',ls='--',lw=1.3,label='Comando manuale: 0,9 m/s')
            ax.axhline(vmax,color='#888888',ls=':',lw=1,label='Limite del comando RL: 1,6 m/s')
            ax.set_title(title+'\n'+('Velocità longitudinale comandata' if column==0 else 'Velocità orizzontale stimata dalle posizioni'))
            ax.set_ylabel('Velocità [m/s]');ax.grid(alpha=.2)
            ax.set_ylim(0, max(1.8,max(actual)*1.05))
            if i==1:ax.set_xlabel('Tempo simulato dall’inizio episodio [s]')
        values=list(episodes.values())
        stats.append(dict(layout=summary['layout'],source=str(directory),checkpoint=str(checkpoint),
            episodes=len(values),command_mean_m_s=float(np.mean(commands)),
            command_min_m_s=float(np.min(commands)),command_max_m_s=float(np.max(commands)),
            command_percent_over_manual=float(100*np.mean(np.array(commands)>.9)),
            measured_horizontal_mean_from_intervals_m_s=float(np.mean(actual)),
            measured_horizontal_max_interval_m_s=float(np.max(actual)),
            full_episode_path_over_time_m_s=float(sum(r['path_m'] for r in values)/sum(r['steps']*dt for r in values)),
            mean_episode_seconds=float(np.mean([r['steps']*dt for r in values])),
            mean_energy_j=float(np.mean([r['energy_j'] for r in values]))))
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Test 12 — policy finale, cinque episodi per mappa, tutti sul percorso asfaltato',fontsize=13)
    fig.text(.5,.015,'Linee sottili: singoli episodi. Velocità effettiva: |Δposizione XY| / 0,2 s; ultimo intervallo non registrato.\n'
             'La linea manuale è un comando di riferimento, non una misura della velocità durante la guida manuale.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95))
    fig.savefig(args.output/'speed.png',dpi=180)
    fig.savefig(args.output/'speed.pdf')
    (args.output/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    with (args.output/'speed_samples.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
