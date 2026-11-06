"""Frozen DINO + frozen PCA BEFORE local-grid insertion; dense RGB-D geometry."""
import time

import cv2
import numpy as np
import torch

from train_general_big_map_test10_SAC.config import DEPTH_STRIDE, DINO_INPUT_SIZE, DINO_MODEL, LOCAL_DINO
from train_general_big_map_test10_SAC.dino_feature_extractor import DinoFeatureExtractor
from train_general_big_map_test10_SAC.perception import backproject, world_to_local, local_cells, cell_indices
from train_general_big_map_test10_SAC.memory import MapBank


class LocalVision:
    def __init__(self, device, pca=None, batch_size=4, model_path=None, capture=False, map_backend='cpu', rgb_backend='opencv'):
        source = str(model_path or (LOCAL_DINO if LOCAL_DINO.is_dir() else DINO_MODEL))
        self.extractor = DinoFeatureExtractor(source, device, DINO_INPUT_SIZE)
        self.pca = pca
        self.batch_size = batch_size
        self.capture = capture
        if map_backend not in ('cpu', 'cuda'):
            raise ValueError('Unknown map backend')
        # Calibration deliberately retains the original capture/preprocessing path.
        self.gpu_maps = map_backend == 'cuda' and not capture
        if self.gpu_maps:
            if torch.device(device).type != 'cuda':
                raise ValueError('CUDA maps require a CUDA simulation device')
            from train_general_big_map_test10_SAC.gpu_memory import TorchMapBank
            self.memory = TorchMapBank(device)
        else:
            self.memory = MapBank()
        if rgb_backend not in ('opencv', 'cuda') or (rgb_backend == 'cuda' and not self.gpu_maps):
            raise ValueError('CUDA RGB requires CUDA maps and no calibration capture')
        self.device_rgb = None
        if rgb_backend == 'cuda':
            from train_general_big_map_test10_SAC.fast_rgb import DeviceRGB
            self.device_rgb = DeviceRGB(self.extractor.processor, device, DINO_INPUT_SIZE)
        from train_general_big_map_test10_SAC.timing import PhaseTimers
        self.timers = PhaseTimers(False, device)
        print(f'[MAP] backend={"cuda" if self.gpu_maps else "cpu"}; RGB preprocessing={rgb_backend}', flush=True)
        self.samples = []
        self.last_ms = 0.0
        if self.extractor.feature_dim != 384:
            raise ValueError('Expected DINOv2-small with 384-dimensional patch tokens')
        self.mean = None if pca is None else torch.tensor(pca.mean, device=device)
        self.components = None if pca is None else torch.tensor(pca.components, device=device)

    def reset(self, env_ids):
        self.memory.reset(env_ids if self.gpu_maps else env_ids.detach().cpu().tolist())

    def extract(self, camera, base_pos, yaw, body_pos, episode_times):
        if self.gpu_maps:
            return self._extract_gpu(camera, base_pos, yaw, body_pos, episode_times)
        start = time.perf_counter()
        data = camera.data
        rgb = data.output['rgb'][..., :3].cpu().numpy()
        depth = data.output['distance_to_image_plane'].cpu().numpy().reshape(len(rgb), 360, 640)
        intrinsics = data.intrinsic_matrices.cpu().numpy()
        positions, rotations = data.pos_w.cpu().numpy(), data.quat_w_ros.cpu().numpy()
        bases, headings, bodies = base_pos.cpu().numpy(), yaw.cpu().numpy(), body_pos.cpu().numpy()
        times = episode_times.cpu().numpy()
        features = []
        raw_features = []
        model = self.extractor
        for first in range(0, len(rgb), self.batch_size):
            chunk = [cv2.resize(im, (518, 518), interpolation=cv2.INTER_LINEAR)
                     for im in rgb[first:first+self.batch_size]]
            inputs = model.processor(images=chunk, return_tensors='pt', do_resize=False, do_center_crop=False)
            with torch.no_grad():
                raw = model.model(pixel_values=inputs['pixel_values'].to(model.device)).last_hidden_state[:, 1:, :]
                compressed = (raw-self.mean) @ self.components.T if self.mean is not None else raw[..., :32]*0
            features.extend(compressed.float().cpu().numpy())
            if self.capture:
                raw_features.extend(raw.float().cpu().numpy())
        result = []
        self.samples = []
        for i in range(len(rgb)):
            patch_w, accepted = backproject(depth[i], intrinsics[i], positions[i], rotations[i], patches=True)
            dense_w, _ = backproject(depth[i], intrinsics[i], positions[i], rotations[i], stride=DEPTH_STRIDE)
            # Conservative self-return rejection using measured link positions; no terrain labels.
            def not_self(points):
                keep = np.ones(len(points), dtype=bool)
                for center in bodies[i]:
                    keep &= np.sum((points-center)**2, axis=1) > 0.08**2
                return keep
            keep = not_self(patch_w)
            patch_w, accepted = patch_w[keep], accepted[keep]
            dense_w = dense_w[not_self(dense_w)]
            patch = world_to_local(patch_w, bases[i], headings[i])
            dense = world_to_local(dense_w, bases[i], headings[i])
            result.append(self.memory.observe(i, patch_w, features[i][accepted], dense_w,
                                              bases[i], headings[i], times[i]))
            if self.capture:
                local = cell_indices(patch) >= 0
                self.samples.append((patch_w[local], raw_features[i][accepted[local]]))
        self.last_ms = (time.perf_counter()-start)*1000
        return torch.as_tensor(np.stack(result), device=base_pos.device)

    @torch.no_grad()
    def _extract_gpu(self, camera, base_pos, yaw, body_pos, episode_times):
        from train_general_big_map_test10_SAC.gpu_memory import backproject_batch
        start = time.perf_counter()
        data = camera.data
        # Device RGB uses the same fixed-point interpolation and processor LUT.
        rgb = data.output['rgb'][..., :3]
        if self.device_rgb is not None:
            if not self.device_rgb.verified_real:
                report = self.device_rgb.verify(rgb)
                print(f'[RGB] Real-frame equivalence PASSED: {report}', flush=True)
        else:
            with self.timers.measure('rgb_preprocess'):
                rgb = rgb.cpu().numpy()
        features = []
        model = self.extractor
        for first in range(0, len(rgb), self.batch_size):
            with self.timers.measure('rgb_preprocess'):
                if self.device_rgb is not None:
                    pixels = self.device_rgb(rgb[first:first+self.batch_size])
                else:
                    chunk = [cv2.resize(im, (518, 518), interpolation=cv2.INTER_LINEAR)
                             for im in rgb[first:first+self.batch_size]]
                    pixels = model.processor(images=chunk, return_tensors='pt', do_resize=False,
                                             do_center_crop=False)['pixel_values'].to(model.device)
            with self.timers.measure('dino_pca'):
                raw = model.model(pixel_values=pixels).last_hidden_state[:, 1:, :]
                compressed = (raw-self.mean) @ self.components.T if self.mean is not None else raw[..., :32]*0
            features.append(compressed.float())
        features = torch.cat(features)
        depth = data.output['distance_to_image_plane'].reshape(len(rgb), 360, 640)
        args = (depth, data.intrinsic_matrices, data.pos_w, data.quat_w_ros, body_pos)
        with self.timers.measure('geometry_map'):
            patches, patch_envs, patch_ids = backproject_batch(*args, patches=True)
            dense, dense_envs, _ = backproject_batch(*args)
            result = self.memory.observe(patches, features[patch_envs, patch_ids], patch_envs,
                                         dense, dense_envs, base_pos, yaw, episode_times)
        self.samples = []
        self.last_ms = (time.perf_counter()-start)*1000
        return result
