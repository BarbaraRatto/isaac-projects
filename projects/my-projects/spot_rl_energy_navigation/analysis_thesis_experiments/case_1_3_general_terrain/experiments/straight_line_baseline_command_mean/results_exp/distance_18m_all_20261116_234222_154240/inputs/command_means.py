"""Time-weighted command means from the original policy trajectories."""
import copy
import csv
import hashlib
import math
from pathlib import Path


def trajectory_groups(path):
    groups={}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(int(row['episode']),[]).append(row)
    return groups


def extract(rows):
    intervals=[]
    for previous,row in zip(rows,rows[1:]):
        dt=float(row['time_s'])-float(previous['time_s'])
        if dt<=0 or int(row['step'])!=int(previous['step'])+1:
            raise ValueError('Nonconsecutive trajectory samples')
        # The row at the end of a step stores the command applied during that step.
        intervals.append(dict(end_time_s=float(row['time_s']),dt_s=dt,phase=row['phase'],
            forward_m_s=float(row['command_vx_m_s']),yaw_rad_s=float(row['command_wz_rad_s'])))
    phases={phase:[r for r in intervals if r['phase']==phase] for phase in ('alignment','navigation')}
    if any(not rows for rows in phases.values()):
        raise ValueError('Both phases require recorded command intervals')
    def average(phase,key,absolute=False):
        return math.fsum(r['dt_s']*(abs(r[key]) if absolute else r[key]) for r in phases[phase])/math.fsum(r['dt_s'] for r in phases[phase])
    if any(abs(r['forward_m_s'])>1e-6 for r in phases['alignment']):
        raise ValueError('Expected zero forward command during original alignment')
    result=dict(navigation_forward_m_s=average('navigation','forward_m_s'),
        alignment_yaw_rad_s=average('alignment','yaw_rad_s'),
        alignment_abs_yaw_rad_s=average('alignment','yaw_rad_s',True),
        alignment_duration_s=math.fsum(r['dt_s'] for r in phases['alignment']),
        navigation_duration_s=math.fsum(r['dt_s'] for r in phases['navigation']),
        sample_count=len(intervals),samples=intervals)
    if not all(math.isfinite(result[key]) for key in ('navigation_forward_m_s','alignment_yaw_rad_s')):
        raise ValueError('Invalid command mean')
    return result


def add_command_means(document,source):
    document=copy.deepcopy(document)
    cache={}
    for case in document['cases']:
        path=Path(case['reference']['source_run'])/'trajectories.csv'
        if path not in cache:
            cache[path]=(trajectory_groups(path),hashlib.sha256(path.read_bytes()).hexdigest())
        groups,digest=cache[path]
        means=extract(groups[case['reference']['source_episode']])
        means.update(source_trajectories=str(path),source_sha256=digest,
            method='Time-weighted commands by phase; exclude the initial step-zero sample')
        case['policy_effective_mean_speed_m_s']=case['reference_speed_m_s']
        case['command_means']=means
        case['reference_speed_m_s']=means['navigation_forward_m_s']
    document['name']=f"Straight-line baseline with phase-specific mean commands, {document['distance_m']} m"
    document['controller'].update(forward_speed='Per-mission time-weighted mean commanded forward speed during original navigation',
        alignment_yaw='Magnitude of signed mean yaw command during original alignment; direction follows heading error; zero within original tolerance')
    document['command_mean_source_panel']=dict(path=str(source),sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    return document


def validate_command_means(document):
    cache={}
    for case in document['cases']:
        saved=case['command_means'];path=Path(saved['source_trajectories'])
        if path not in cache:
            cache[path]=(trajectory_groups(path),hashlib.sha256(path.read_bytes()).hexdigest())
        groups,digest=cache[path]
        if digest!=saved['source_sha256']:
            raise ValueError('Original command log changed')
        actual=extract(groups[case['reference']['source_episode']])
        for key,value in actual.items():
            if isinstance(value,(int,float)):
                if not math.isclose(saved[key],value,rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Recorded command mean differs')
            elif saved[key]!=value:
                raise ValueError('Recorded command samples differ')
        if abs(saved['alignment_yaw_rad_s'])>0.800001:
            raise ValueError('Alignment command exceeds original limit')
