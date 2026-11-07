"""Select a completed training's best, including inherited best after resume."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def select_best(runs_root):
    root=Path(runs_root).resolve()
    finals=list(root.glob('*/policy_final.pt'))
    if not finals:
        raise FileNotFoundError('Nessun training completato: indica --checkpoint o attendi la fine.')
    run=max(finals,key=lambda p:p.stat().st_mtime).parent
    seen=set()
    while run not in seen:
        seen.add(run)
        best=run/'policy_best.pt'
        if best.is_file():
            return best
        config_path=run/'config.json'
        if not config_path.is_file():
            break
        config=json.loads(config_path.read_text())
        resume=config.get('arguments',{}).get('resume')
        if not resume or not config.get('initial_transitions',0):
            break
        checkpoint=Path(resume)
        candidates=[checkpoint, HERE.parents[1]/checkpoint,
                    root/checkpoint.parent.name/checkpoint.name]
        parent=next((p.resolve().parent for p in candidates if p.is_file()),None)
        if parent is None or not (parent/'config.json').is_file():
            break
        previous=json.loads((parent/'config.json').read_text())
        if any(config.get(key)!=previous.get(key) for key in ('signature','identity')):
            raise ValueError('La run precedente ha configurazione o asset diversi.')
        run=parent
    raise FileNotFoundError('Policy best non trovata nella run completata o nella sua catena di ripresa.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path,default=HERE/'runs')
    args=parser.parse_args()
    print(select_best(args.runs_root))


if __name__=='__main__':main()
