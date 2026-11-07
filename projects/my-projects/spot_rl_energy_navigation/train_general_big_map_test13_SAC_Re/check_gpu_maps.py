"""Standalone numerical parity checks; no Isaac Sim, assets or DINO required.

Run with --device cuda:0 on the training host before enabling CUDA maps.
This validates the CUDA arithmetic against the CPU reference; it is not a
simulator benchmark or a guarantee of whole-training throughput.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test13_SAC_Re.memory import MapBank
from train_general_big_map_test13_SAC_Re.gpu_memory import TorchMapBank, backproject_batch
from train_general_big_map_test13_SAC_Re.perception import backproject


def check_maps(device):
    rng = np.random.default_rng(4701)
    cpu, gpu = MapBank(), TorchMapBank(device)
    tensor = lambda x: torch.as_tensor(x, device=device)
    n = 4
    origins = np.array([[0, 0, .5], [8000, 16000, .5], [-16000, 8000, .5], [24000, -8000, .5]], dtype=float)
    for step in range(24):
        reset = [1, 3] if step in (7, 17) else []
        if step == 12:
            reset = list(range(n))
        cpu.reset(reset); gpu.reset(tensor(reset).long())
        bases = origins + [step*.12, .1*np.sin(step), .05*np.sin(step)]
        yaw = np.arange(n)*.37 + step*.12
        times = np.full(n, step*.2)
        patches, features, dense, pe, de, expected = [], [], [], [], [], []
        for env in range(n):
            pts = rng.uniform([-2, -2, -.03], [2, 2, .03], (1200, 3))
            pts[:, 2] += .2*pts[:, 0]-.1*pts[:, 1]
            pts += bases[env]
            patch = pts[:120].copy()
            values = rng.normal(size=(len(patch),32)).astype(np.float32)
            if step%5 == 0:
                patch, values = patch[:0], values[:0]
            if step%7 == 0:
                pts = pts[:0]
            patches.append(patch); features.append(values); dense.append(pts)
            pe.append(np.full(len(patch),env,dtype=np.int64)); de.append(np.full(len(pts),env,dtype=np.int64))
            expected.append(cpu.observe(env,patch,values,pts,bases[env],yaw[env],times[env]))
        actual = gpu.observe(*[tensor(np.concatenate(x)) for x in (patches,features,pe,dense,de)],
                             tensor(bases),tensor(yaw),tensor(times)).cpu().numpy()
        expected = np.stack(expected)
        np.testing.assert_array_equal(actual[:,:,36:38],expected[:,:,36:38])
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    # Query without observations: ages saturate and previously seen bins persist.
    times += 200
    expected = np.stack([cpu.maps[i].query(bases[i],yaw[i],times[i]) for i in range(n)])
    np.testing.assert_allclose(gpu.query(tensor(bases),tensor(yaw),tensor(times)).cpu(),expected,rtol=2e-5,atol=2e-6)
    gpu.reset(tensor(np.arange(n)))
    assert not gpu.query(tensor(bases),tensor(yaw),tensor(times)).any()
    print('[OK] Map outputs/masks, motion, slopes, aging, large origins and partial/full resets.',flush=True)


def check_backprojection(device):
    rng = np.random.default_rng(11)
    n, h, w = 3, 48, 64
    depth = rng.uniform(.1, 16, (n,h,w)).astype(np.float32)
    depth[:,::4,::4] = np.nan
    depth[:,1,1] = np.inf
    depth[:,2,2] = .2
    depth[:,3,3] = 15.
    k = np.tile([[45.,0,32],[0,45,24],[0,0,1]],(n,1,1)).astype(np.float32)
    positions = np.array([[0,0,.5],[8000,16000,.5],[-8000,16000,.5]],dtype=np.float32)
    quats = rng.normal(size=(n,4)); quats/=np.linalg.norm(quats,axis=1)[:,None]
    quats = quats.astype(np.float32)
    bodies = np.repeat(positions[:,None,:],3,axis=1)
    # Put a body exactly on a valid sampled point to exercise self rejection.
    for i in range(n):
        points,_ = backproject(depth[i],k[i],positions[i],quats[i],patches=True)
        bodies[i,0] = points[0]
    args = [torch.as_tensor(x,device=device) for x in (depth,k,positions,quats,bodies)]
    for patches in (False,True):
        got,envs,ids = backproject_batch(*args,patches=patches)
        expected,ei,pi = [],[],[]
        for i in range(n):
            points,indices = backproject(depth[i],k[i],positions[i],quats[i],patches=patches)
            keep = np.ones(len(points),dtype=bool)
            for body in bodies[i]:
                keep &= ((points-body)**2).sum(1) > .08**2
            expected.append(points[keep]); ei.append(np.full(keep.sum(),i)); pi.append(indices[keep])
        np.testing.assert_allclose(got.cpu(),np.concatenate(expected),rtol=1e-10,atol=1e-9)
        np.testing.assert_array_equal(envs.cpu(),np.concatenate(ei))
        np.testing.assert_array_equal(ids.cpu(),np.concatenate(pi))
    print('[OK] RGB patch/depth sampling, invalid depths, ROS rotations and robot-point rejection.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run this check on the training GPU')
    torch.set_num_threads(4)
    start=time.perf_counter()
    check_backprojection(device)
    check_maps(device)
    print(f'[OK] Numerical checks passed on {device} in {time.perf_counter()-start:.1f}s.',flush=True)


if __name__=='__main__':
    main()
