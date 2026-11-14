"""Update collection reviews without deleting original measurements."""
import argparse
import copy
from pathlib import Path
from add_episode import HERE, REGISTRY, read, write, rebuild


def remove_episode(episode,reason):
    document=read(REGISTRY)
    slot=next(s for s in document['slots'] if s['episode']==episode)
    if slot['status']!='selected':
        raise ValueError('Episode is not selected')
    previous=copy.deepcopy(slot)
    source=HERE/slot['episode_directory']
    destination=HERE/'excluded'/f"episode_{episode:02d}_source_{slot['source_episode']}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(exist_ok=True)
    source.rename(destination)
    previous.update(episode_directory=str(destination.relative_to(HERE)),exclusion_reason=reason)
    document.setdefault('removed_selections',[]).append(previous)
    slot.clear()
    slot.update(episode=episode,status='pending')
    write(destination/'exclusion.json',dict(collection_episode=episode,reason=reason,source_episode=previous['source_episode'],
                                           source_run=previous['source_run'],slot_is_available=True))
    write(REGISTRY,document)
    rebuild(document)
    print(f'E{episode} excluded; archived data retained.')


def set_review(episode,status):
    document=read(REGISTRY)
    slot=next(s for s in document['slots'] if s['episode']==episode)
    if slot['status']!='selected':
        raise ValueError('Episode is not selected')
    slot['review_status']=status
    slot['case']['review_status']=status
    write(HERE/slot['episode_directory']/'mission.json',slot['case'])
    provenance_path=HERE/slot['episode_directory']/'provenance.json'
    provenance=read(provenance_path)
    provenance['review_status']=status
    write(provenance_path,provenance)
    write(REGISTRY,document)
    rebuild(document)
    print(f'E{episode}: {status}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=int)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--remove',action='store_true')
    group.add_argument('--review',choices=('confirmed','tentative'))
    parser.add_argument('--reason',default='Removed from the illustrative collection by user request')
    args=parser.parse_args()
    if args.remove:
        remove_episode(args.episode,args.reason)
    else:
        set_review(args.episode,args.review)
