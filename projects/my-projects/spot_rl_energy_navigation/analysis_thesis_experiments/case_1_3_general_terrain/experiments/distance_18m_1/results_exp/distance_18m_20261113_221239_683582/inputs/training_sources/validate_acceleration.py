"""No simulator/training: validate accelerated RGB and legacy full resume on this host."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test9_SAC.config import LOCAL_DINO, OBSERVATION_SIZE, SAC_CFG
from train_general_big_map_test9_SAC.fast_rgb import DeviceRGB
from train_general_big_map_test9_SAC.replay import ReplayBuffer
from train_general_big_map_test9_SAC.sac import SAC
from train_general_big_map_test9_SAC.runtime_config import runtime_signature


def validate(device, model_path=LOCAL_DINO, resume=None, check_dino=True):
    from transformers import AutoImageProcessor, AutoModel
    torch.set_num_threads(4)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; validate with the cloud Python environment after stopping the old simulator')
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    fast = DeviceRGB(processor, device)
    rng = np.random.default_rng(42)
    images = rng.integers(0, 256, (2, 360, 640, 3), dtype=np.uint8)
    # Constant colors and sharp edges also exercise rounding and border handling.
    patterns = np.zeros_like(images)
    patterns[0, ::2, ::2] = 255
    patterns[1] = [0, 127, 255]
    report = dict(device=str(device), torch=torch.__version__, opencv=cv2.__version__)
    for rgb in (images, patterns):
        report['rgb_equivalence'] = fast.verify(torch.as_tensor(rgb, device=device))
    tensor = torch.as_tensor(images, device=device)
    def reference():
        chunk = [cv2.resize(im, (518, 518), interpolation=cv2.INTER_LINEAR) for im in tensor.cpu().numpy()]
        return processor(images=chunk, return_tensors='pt', do_resize=False,
                         do_center_crop=False)['pixel_values'].to(device)
    old, new = reference(), fast(tensor)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def benchmark(function):
        function()
        synchronize()
        start = time.perf_counter()
        for _ in range(3):
            function()
        synchronize()
        return 1000*(time.perf_counter()-start)/3
    report['rgb_two_frames_ms'] = dict(opencv=benchmark(reference), device=benchmark(lambda: fast(tensor)))
    if check_dino:
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
        with torch.no_grad():
            before = model(pixel_values=old).last_hidden_state
            after = model(pixel_values=new).last_hidden_state
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)
        report['dino_max_error'] = float((before-after).abs().max())
        del model, before, after
    # Old .npz schema -> tensor replay -> new snapshot -> CPU reader.
    with tempfile.TemporaryDirectory(prefix='test9-replay-check-') as directory:
        original = ReplayBuffer(7)
        a = torch.zeros(4, OBSERVATION_SIZE)
        c = torch.cat((a, torch.ones(4, 24)), 1)
        original.add(a, c, torch.zeros(4, 2), torch.ones(4), torch.zeros(4, dtype=torch.bool), a, c)
        legacy = Path(directory)/'old.npz'
        np.savez(legacy, capacity=7, size=4, position=4, total=4, **{k: v[:4] for k, v in original.arrays.items()})
        accelerated = ReplayBuffer(7, device=device, tensor_storage=True)
        accelerated.load(legacy)
        accelerated.sample(3, device)
        accelerated.save(Path(directory)/'new.npz')
        original.load(Path(directory)/'new.npz')
        for key in original.arrays:
            np.testing.assert_array_equal(original.arrays[key][:4], accelerated.arrays[key][:4].cpu().numpy())
    report['legacy_replay_roundtrip'] = True
    if resume is not None:
        from train_general_big_map_test9_SAC.recovery import resolve_resume
        from train_general_big_map_test9_SAC.artifacts import source_identity
        checkpoint = resolve_resume(resume)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        settings = replace(SAC_CFG, **state['learner']['settings'])
        if state['signature'] != runtime_signature(settings) or state['identity'] != source_identity(model_path):
            raise ValueError('Existing checkpoint is incompatible with experiment/assets')
        learner = SAC(device, settings=settings)
        learner.load_state_dict(state['learner'])
        replay = ReplayBuffer(settings.buffer_size, device=device)
        info = state['training_state']['replay_snapshot']
        replay.load(checkpoint.parent/info['file'])
        if replay.total != learner.transitions or info['transitions'] != learner.transitions:
            raise ValueError('Actual replay/learner counters differ')
        if not replay.size:
            raise ValueError('Cannot compare policy on an empty actual replay')
        a = torch.as_tensor(replay.arrays['actor'][:4], device=device)
        priv = torch.as_tensor(replay.arrays['privileged'][:4], device=device)
        cpu = SAC('cpu', settings=settings)
        cpu.load_state_dict(state['learner'])
        with torch.no_grad():
            actual = learner.policy.act(a, torch.cat((a, priv), 1), deterministic=True)[0].cpu()
            expected = cpu.policy.act(a.cpu(), torch.cat((a, priv), 1).cpu(), deterministic=True)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        report['resume'] = dict(checkpoint=str(checkpoint), transitions=learner.transitions,
                              updates=learner.updates, replay_size=replay.size,
                              curriculum=state['training_state']['curriculum'],
                              policy_cpu_device_max_error=float((actual-expected).abs().max()))
    report['passed'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = validate(args.device, args.dino_model, args.resume)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print('[ACCELERATION] PASS: no simulation, no learner updates, no existing checkpoints modified.')


if __name__ == '__main__':
    main()
