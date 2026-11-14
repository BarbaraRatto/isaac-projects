"""Archive a manually selected episode while keeping its original provenance."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
CASE=HERE.parents[1]
PROJECT=HERE.parents[3]
REGISTRY=HERE/'mission_list.json'
MODEL=PROJECT/'train_general_big_map_test7_SAC/checkpoints_importati/policy_best_stage_3.pt'
MAP=CASE/'maps/real_terrains_v2_eval_01.usd'
RAW_FILES=('episodes.csv','trajectories.csv','power_samples.csv','fall_diagnostics.csv')
EPISODE_CACHE={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,data):
    temporary=Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def prime_cache(source,episode_ids):
    """Read a large source log once when collecting several episodes."""
    source=Path(source).resolve()
    wanted=set(episode_ids)
    for name in RAW_FILES:
        grouped={episode:[] for episode in wanted}
        with (source/name).open(newline='') as stream:
            for row in csv.DictReader(stream):
                episode=int(row['episode'])
                if episode in wanted:
                    grouped[episode].append(row)
        for episode,data in grouped.items():
            EPISODE_CACHE[str(source/name),episode]=data


def episode_rows(path,episode):
    key=str(Path(path).resolve()),episode
    if key in EPISODE_CACHE:
        return EPISODE_CACHE[key]
    with Path(path).open(newline='') as stream:
        return [r for r in csv.DictReader(stream) if int(r['episode'])==episode]


def csv_write(path,data,fields=None):
    if fields is None:
        fields=list(data[0])
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def initialize():
    if REGISTRY.exists():
        return
    write(REGISTRY,dict(schema=1,name='Selected illustrative twelve-metre episodes',
        selection_method='Manual outcome-based selection across separate complete evaluation runs',
        interpretation='Illustrative collection; not an unbiased random test sample or a success-rate estimate',
        target_count=40,category_targets={name:10 for name in ('regular','ice','stairs','surface_choice')},
        checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),map_sha256=sha(MAP),
        slots=[dict(episode=i,status='pending') for i in range(1,41)]))


def rebuild(document):
    fields=['episode','status','review_status','selection_note','category','start_x_m','start_y_m','goal_x_m','goal_y_m',
            'direct_distance_m','source_run','source_episode','source_episode_12m','reason','energy_j','path_m','time_s']
    table=[]; cases=[]
    selected=[s for s in document['slots'] if s['status']=='selected']
    for slot in document['slots']:
        row={key:'' for key in fields}
        row.update(episode=slot['episode'],status=slot['status'])
        if slot['status']=='selected':
            case=slot['case']; m=case['mission']; r=slot['result']
            row.update(selection_note=slot.get('selection_note',''),review_status=slot.get('review_status','confirmed'),category=case['category'],start_x_m=m['start'][0],start_y_m=m['start'][1],
                goal_x_m=m['goal'][0],goal_y_m=m['goal'][1],direct_distance_m=m['distance_m'],
                source_run=slot['source_run'],source_episode=slot['source_episode'],
                source_episode_12m=slot['source_episode_12m'],reason=r['reason'],energy_j=r['energy_j'],
                path_m=r['path_m'],time_s=r['time_s'])
            cases.append(case)
        table.append(row)
    csv_write(HERE/'mission_list.csv',table,fields)
    summary=[]
    for category,target in document['category_targets'].items():
        subset=[s for s in selected if s['case']['category']==category]
        tentative=sum(s.get('review_status','confirmed')=='tentative' for s in subset)
        summary.append(dict(category=category,saved=len(subset),confirmed=len(subset)-tentative,
                            tentative=tentative,target=target,remaining=max(0,target-len(subset)),over_target=max(0,len(subset)-target)))
    csv_write(HERE/'category_counts.csv',summary)
    write(HERE/'category_summary.json',dict(total_saved=len(selected),
        total_confirmed=sum(s['confirmed'] for s in summary),total_tentative=sum(s['tentative'] for s in summary),
        remaining_slots=40-len(selected),over_target_count=sum(s['over_target'] for s in summary),
        needed_to_fill_categories=sum(s['remaining'] for s in summary),categories=summary))
    write(HERE/'selected_missions.json',dict(name=document['name'],selection_method=document['selection_method'],
        map_sha256=document['map_sha256'],target_count=40,cases=cases))
    for name in RAW_FILES:
        data=[]
        for slot in selected:
            data.extend(rows(HERE/slot['episode_directory']/name))
        if data:
            csv_write(HERE/name,data)


def add_episode(source,source_episode,slot_number=None,tentative=False,note='',allow_category_overflow=False):
    source=Path(source).resolve()
    if not (source/'complete.json').is_file():
        raise ValueError('Archive only completed runs with logging checks')
    config=read(source/'evaluation_config.json')
    document=read(REGISTRY)
    if config['checkpoint_sha256']!=document['checkpoint_sha256'] or config['map_sha256']!=document['map_sha256']:
        raise ValueError('Selected run has a different checkpoint or map')
    if any(s.get('source_run')==str(source) and s.get('source_episode')==source_episode for s in document['slots']):
        raise ValueError('Episode already collected')
    subset=read(source/'inputs'/config['mission_file'])
    source_case=next(c for c in subset['cases'] if c['episode']==source_episode)
    if source_case['distance_group_m']!=12:
        raise ValueError('Select an twelve-metre episode')
    category=source_case['category']
    if not allow_category_overflow and sum(s.get('case',{}).get('category')==category for s in document['slots'])>=document['category_targets'][category]:
        raise ValueError('Category quota already full')
    available=[s for s in document['slots'] if s['status']=='pending' and (slot_number is None or s['episode']==slot_number)]
    if not available:
        raise ValueError('No free requested slot')
    selected_slot=available[0]
    new_id=selected_slot['episode']
    source_result=episode_rows(source/'episodes.csv',source_episode)[0]
    relative=Path('episodes')/f'episode_{new_id:02d}'
    target=HERE/relative
    if target.exists():
        raise FileExistsError(target)
    selected_data={}
    for name in RAW_FILES:
        selected_data[name]=episode_rows(source/name,source_episode)
        if not selected_data[name]:
            raise ValueError(f'Missing episode data in {name}')
    trajectory=selected_data['trajectories.csv']
    if int(trajectory[-1]['terminal'])!=1 or int(trajectory[-1]['step'])!=int(source_result['steps']):
        raise ValueError('Missing terminal trajectory state')
    if not math.isclose(float(trajectory[-1]['energy_j']),float(source_result['energy_j']),abs_tol=.001):
        raise ValueError('Terminal energy mismatch')
    source_key=source.name+'_'+sha(source/'evaluation_config.json')[:8]
    archive=HERE/'sources'/source_key
    if not archive.exists():
        archive.mkdir(parents=True)
        shutil.copytree(source/'inputs',archive/'inputs')
        for name in ('evaluation_config.json','summary.json','complete.json'):
            shutil.copy2(source/name,archive/name)
    target.mkdir(parents=True)
    long_ids=[c['episode'] for c in subset['cases'] if c['distance_group_m']==12]
    local_id=source_episode
    for name,data in selected_data.items():
        converted=[dict(r,episode=new_id,source_episode=source_episode,source_episode_12m=local_id,source_run=source_key) for r in data]
        csv_write(target/name,converted)
    case=copy.deepcopy(source_case)
    case.update(selection_note=note,review_status='tentative' if tentative else 'confirmed',episode=new_id,source_episode=source_episode,source_episode_12m=local_id,source_run=source_key)
    result=dict(source_result,episode=new_id,source_episode=source_episode,source_episode_12m=local_id,source_run=source_key)
    write(target/'mission.json',case)
    write(target/'source_result.json',source_result)
    provenance=dict(source_run=str(source),source_episode=source_episode,source_episode_12m=local_id,
        collection_episode=new_id,selection_note=note,review_status='tentative' if tentative else 'confirmed',source_config_sha256=sha(source/'evaluation_config.json'),
        source_raw_sha256={name:sha(source/name) for name in RAW_FILES},
        archived_inputs=str((Path('sources')/source_key/'inputs')),
        checkpoint_sha256=config['checkpoint_sha256'],map_sha256=config['map_sha256'],
        selection='Manually requested illustrative episode; original run retained separately')
    write(target/'provenance.json',provenance)
    selected_slot.update(status='selected',selection_note=note,review_status='tentative' if tentative else 'confirmed',source_run=str(source),source_episode=source_episode,
        source_episode_12m=local_id,episode_directory=str(relative),case=case,result=result)
    write(REGISTRY,document)
    rebuild(document)
    print(f'E{new_id} <- source Ep {source_episode} (12 m index {local_id}), {category}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--source-episode',type=int,required=True)
    parser.add_argument('--slot',type=int)
    parser.add_argument('--note',default='')
    parser.add_argument('--allow-category-overflow',action='store_true')
    parser.add_argument('--tentative',action='store_true',help='Archive provisionally, pending confirmation')
    args=parser.parse_args()
    initialize()
    add_episode(args.source_run,args.source_episode,args.slot,args.tentative,args.note,args.allow_category_overflow)
