"""One-time RGBD final-cell calibration, no RL updates and no PCA refit.

Run by train.sh ONLY when scalers are missing. Capture from safe TRAIN poses;
independent calibration hold-out positions never overlap fit positions. Ice and
stairs are observed from safe neighboring material, never used for spawn.
"""
import argparse
from pathlib import Path
import sys
import json
import math

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test10_SAC.config import DEFAULT_PCA,LOCAL_DINO,LOOK_X_M,LOOK_Y_M


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--dino-batch-size',type=int,default=32)
    parser.add_argument('--dino-model',type=Path,default=LOCAL_DINO)
    parser.add_argument('--pca',type=Path,default=DEFAULT_PCA)
    parser.add_argument('--train-views-per-material',type=int,default=512)
    parser.add_argument('--validation-views-per-material',type=int,default=128)
    parser.add_argument('--output',type=Path,default=HERE/'artifacts/frozen_scalers.json')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();args.enable_cameras=True
    if min(args.num_envs,args.dino_batch_size,args.train_views_per_material,args.validation_views_per_material)<1:
        parser.error('Positive counts required')
    if args.output.exists():raise FileExistsError('Calibration is immutable; refusing to overwrite '+str(args.output))
    import numpy as np
    import torch
    from train_general_big_map_test10_SAC.assets import validate_assets
    from train_general_big_map_test10_SAC.artifacts import source_identity
    from train_general_big_map_test10_SAC.calibration import view_plan,point_labels,validate_calibration
    from train_general_big_map_test10_SAC.frozen_scalers import validate_scalers
    from train_general_big_map_test10_SAC.scaler_calibration import CellReservoir,fit_scalers
    from train_general_big_map_test10_SAC.perception import FrozenPCA,file_sha256
    from train_general_big_map_test10_SAC.recovery import atomic_json
    validate_assets();identity=source_identity(args.dino_model)
    pca=FrozenPCA.load(args.pca);validate_calibration(pca,identity)
    with np.load(HERE/'artifacts/spawn_bank.npz') as bank:
        points=bank['points'].copy();evaluation=bank['evaluation'].copy()
    meta=json.loads((HERE/'artifacts/map_metadata.json').read_text())
    plan=view_plan(points,evaluation,meta,seed=123,
                   train_views=args.train_views_per_material,validation_views=args.validation_views_per_material)
    fit_ids={v['point_id'] for v in plan if v['split']=='train'}
    hold_ids={v['point_id'] for v in plan if v['split']=='validation'}
    if fit_ids&hold_ids:raise RuntimeError('Calibration viewpoint leakage')
    reservoirs={'train':CellReservoir(seed=123),'validation':CellReservoir(capacity=1200,seed=456)}
    targets={'train':2000,'validation':500};minimum_views={'train':32,'validation':8}
    used={'train':set(),'validation':set()}
    app=env=None
    try:
        app=AppLauncher(args).app
        from train_general_big_map_test10_SAC.env import GeneralMapEnv
        from train_general_big_map_test10_SAC.scene import GeneralMapCfg
        from train_general_big_map_test10_SAC.layouts import training_conditions
        cfg=GeneralMapCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device
        env=GeneralMapEnv(cfg,training_conditions(args.num_envs),pca=pca,model_path=args.dino_model,
                          dino_batch_size=args.dino_batch_size,map_backend='cuda',rgb_backend='cuda')
        env.reset(seed=123)
        ids=torch.arange(env.num_envs,device=env.device)
        gx,gy=np.meshgrid(LOOK_X_M,LOOK_Y_M,indexing='ij');local=np.stack((gx.ravel(),gy.ravel()),1)
        for first in range(0,len(plan),env.num_envs):
            actual=plan[first:first+env.num_envs]
            if not actual:break
            views=actual+[actual[-1]]*(env.num_envs-len(actual))
            env.vision.reset(ids)
            for turn,angle_offset in enumerate((-.35,0.,.35)):
                root=env.settled_root.clone()
                for i,v in enumerate(views):
                    point=points[v['point_id']];yaw=v['yaw']+angle_offset
                    root[i,:2]=env.scene.env_origins[i,:2]+torch.tensor(point[:2],device=env.device)
                    root[i,2]=env.scene.env_origins[i,2]+float(point[3])+env.standing_clearance[i]+.02
                    root[i,3:7]=torch.tensor([math.cos(yaw/2),0.,0.,math.sin(yaw/2)],device=env.device)
                    root[i,7:]=0
                env.robot.write_root_state_to_sim(root)
                env.robot.write_joint_state_to_sim(env.settled_joint_pos,env.settled_joint_vel)
                env.robot.reset();env.controller.reset(ids)
                for _ in range(40):
                    env.controller.apply(torch.zeros_like(env.command))
                    env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
                env.episode_length_buf[:]=turn
                obs=env._get_observations()['policy']
                cells=obs[:,15:].reshape(env.num_envs,81,40).cpu().numpy()
                bases=(env.robot.data.root_pos_w-env.scene.env_origins).cpu().numpy()
                yaw=env.yaw().cpu().numpy()
                for i,v in enumerate(actual):
                    world=local@np.array([[np.cos(yaw[i]),np.sin(yaw[i])],[-np.sin(yaw[i]),np.cos(yaw[i])]])+bases[i,:2]
                    labels=point_labels(world,meta,margin=.4)
                    labels[labels<0]=7 # observed boundary/mixed cells, never unknown cells
                    reservoirs[v['split']].add(cells[i],labels,v['point_id'])
                    used[v['split']].add(int(v['point_id']))
            if first%(env.num_envs*8)==0:
                print('[CALIBRATION] '+str({k:[len(x) for x in r.values] for k,r in reservoirs.items()}),flush=True)
            if all(r.ready(targets[k],minimum_views[k]) for k,r in reservoirs.items()):break
        for name,r in reservoirs.items():
            if not r.ready(targets[name],minimum_views[name]):
                raise RuntimeError(f'Insufficient {name} final cells/views: '+str([len(x) for x in r.values])+
                                   '; increase calibration views; RL WILL NOT START')
        train=reservoirs['train'].balanced(2000);validation=reservoirs['validation'].balanced(500)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        samples=args.output.with_suffix('.samples.npz')
        temporary=samples.with_suffix('.pending.npz')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,train=train[0],train_labels=train[1],validation=validation[0],validation_labels=validation[1])
        provenance=dict(seed=123,train_viewpoints=sorted(used['train']),validation_viewpoints=sorted(used['validation']),
                        samples_sha256=file_sha256(temporary),samples_file=samples.name,
                        input_pipeline='RGBD -> frozen DINO -> existing PCA32 -> persistent grid -> valid final cells')
        data=fit_scalers(train,validation,identity,file_sha256(args.pca),provenance)
        validate_scalers(data,identity,file_sha256(args.pca))
        temporary.replace(samples)
        atomic_json(args.output,data)
        print('[CALIBRATION] PASS: immutable final-cell scalers saved; no RL updates performed.',flush=True)
    finally:
        if env is not None:env.close()
        if app is not None:app.close(skip_cleanup=args.headless)


if __name__=='__main__':main()
