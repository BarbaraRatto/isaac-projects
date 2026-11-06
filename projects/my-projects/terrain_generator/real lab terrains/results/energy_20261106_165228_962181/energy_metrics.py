"""Physics-step integration and comparison, independent of Isaac Sim."""
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import statistics


@dataclass
class TraversalMeter:
    start_x: float = 1.
    finish_x: float = 7.
    active: bool = False
    completed: bool = False
    energy_j: float = 0.
    seconds: float = 0.
    path_m: float = 0.
    forward_m: float = 0.

    def __post_init__(self):
        if not self.finish_x > self.start_x:
            raise ValueError('Measurement finish must be beyond its start')

    def update(self, previous_xy, xy, previous_power, power, dt):
        if self.completed:return
        x0,y0=previous_xy;x1,y1=xy;dx=x1-x0
        begin=0.
        if not self.active:
            if dx<=0 or x1<self.start_x:return
            begin=max(0.,min(1.,(self.start_x-x0)/dx))
            self.active=True
        end=1.
        if dx>0 and x1>=self.finish_x:
            end=max(begin,min(1.,(self.finish_x-x0)/dx));self.completed=True
        fraction=end-begin
        # Trapezoidal power is linear within each physics step. Integrate only
        # the part between measurement gates, including backtracking once active.
        self.energy_j+=dt*(previous_power*fraction+.5*(power-previous_power)*(end*end-begin*begin))
        self.seconds+=dt*fraction
        self.path_m+=math.hypot(x1-x0,y1-y0)*fraction
        self.forward_m+=dx*fraction


def write_report(directory):
    directory=Path(directory)
    with (directory/'trials.csv').open() as stream:rows=list(csv.DictReader(stream))
    speeds=sorted({float(r['command_speed_m_s']) for r in rows})
    groups=[];pairs=[]
    for speed in speeds:
        for material in ('grey','green'):
            all_rows=[r for r in rows if float(r['command_speed_m_s'])==speed and r['material']==material]
            good=[r for r in all_rows if r['reason']=='completed']
            values=[float(r['j_per_forward_m']) for r in good]
            groups.append(dict(speed=speed,material=material,completed=len(good),attempts=len(all_rows),
                               mean_j_per_forward_m=statistics.mean(values) if values else None,
                               std_j_per_forward_m=statistics.stdev(values) if len(values)>1 else None,
                               mean_actual_speed_m_s=statistics.mean(float(r['mean_forward_speed_m_s']) for r in good) if good else None))
        delta=[];percent=[]
        for trial in sorted({r['repeat'] for r in rows}):
            pair={r['material']:r for r in rows if float(r['command_speed_m_s'])==speed and r['repeat']==trial and r['reason']=='completed'}
            if set(pair)=={'grey','green'}:
                grey=float(pair['grey']['j_per_forward_m']);green=float(pair['green']['j_per_forward_m'])
                delta.append(green-grey)
                if grey>0:percent.append(100*(green/grey-1))
        pairs.append(dict(speed=speed,valid_pairs=len(delta),
                          mean_green_minus_grey_j_per_m=statistics.mean(delta) if delta else None,
                          mean_green_percent_increase=statistics.mean(percent) if percent else None))
    result=dict(groups=groups,paired_comparisons=pairs,
                measurement='Integral of sum(abs(measured_joint_effort * joint_velocity)), fixed forward-distance window',
                limitations='Mechanical joint-work proxy, not battery energy. Failures excluded from energy averages and reported separately; speed and stability must also be compared.')
    (directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    report=['# Confronto energetico grigio/verde','',
            '| Velocità comandata (m/s) | Materiale | Tratti completati | J/m avanzamento, media ± deviazione standard | Velocità effettiva (m/s) |',
            '|---:|---|---:|---:|---:|']
    for g in groups:
        energy='n/d' if g['mean_j_per_forward_m'] is None else f'{g["mean_j_per_forward_m"]:.1f}'
        if g['std_j_per_forward_m'] is not None:energy+=f' ± {g["std_j_per_forward_m"]:.1f}'
        actual='n/d' if g['mean_actual_speed_m_s'] is None else f'{g["mean_actual_speed_m_s"]:.3f}'
        report.append(f'| {g["speed"]:.2f} | {g["material"]} | {g["completed"]}/{g["attempts"]} | {energy} | {actual} |')
    report+=['','| Velocità (m/s) | Coppie completate | Verde − grigio (J/m) | Aumento medio verde (%) |','|---:|---:|---:|---:|']
    for p in pairs:
        delta='n/d' if p['mean_green_minus_grey_j_per_m'] is None else f'{p["mean_green_minus_grey_j_per_m"]:+.1f}'
        percent='n/d' if p['mean_green_percent_increase'] is None else f'{p["mean_green_percent_increase"]:+.1f}%'
        report.append(f'| {p["speed"]:.2f} | {p["valid_pairs"]} | {delta} | {percent} |')
    report+=['','## Interpretazione','',
             '- Misura meccanica dai giunti, non consumo elettrico della batteria.',
             '- J/m di avanzamento usa la stessa distanza longitudinale per entrambi i materiali. trials.csv contiene anche J/m di percorso reale.',
             '- La misura esclude assestamento e rincorsa; include eventuali arresti, oscillazioni o retromarcia fra i due traguardi.',
             '- Confrontare anche velocità effettiva e frazione di tratti completati: un terreno che fa cadere Spot non soddisfa lo scopo del test.',
             '- Il confronto a pari comando non è necessariamente a pari velocità reale. Se differiscono molto, ripetere regolando i comandi.',
             '- Differenze piccole rispetto alla dispersione non dimostrano un costo maggiore. Ripetizioni deterministiche non sono prove statisticamente indipendenti.']
    (directory/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report[:len(groups)+4]),flush=True)
    print('[ENERGY] Report: '+str(directory/'REPORT.md'),flush=True)
    return result
