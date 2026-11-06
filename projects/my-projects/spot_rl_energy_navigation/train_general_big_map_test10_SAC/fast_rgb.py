"""Device-resident equivalent of the existing uint8 OpenCV/BitImageProcessor path.

OpenCV INTER_LINEAR uses 11-bit coefficients and staged integer rounding for
uint8 RGB (opencv 4.11, modules/imgproc/src/resize.cpp). Plain float bilinear
interpolation is NOT the same operation. Coefficients are prepared once on CPU;
images, interpolation and lookup-table normalization stay on the input device.
The installed OpenCV and processor are checked on synthetic AND real frames.
"""
import numpy as np
import torch


class DeviceRGB:
    def __init__(self, processor, device, size=518):
        self.processor, self.device, self.size = processor, torch.device(device), size
        self.axes = {}
        self.verified_real = False
        # A processor-generated LUT preserves its exact rescale/normalize rounding.
        palette = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
        values = processor(images=[palette], return_tensors='pt', do_resize=False,
                           do_center_crop=False)['pixel_values'][0]
        if values.shape != (3, 16, 16):
            raise ValueError('Unsupported image processor for device RGB')
        self.lut = values.reshape(3, 256).to(self.device)

    def _axis(self, source, horizontal):
        key = (source, horizontal)
        if key not in self.axes:
            pos = ((np.arange(self.size)+.5)*(source/self.size)-.5).astype(np.float32)
            index = np.floor(pos).astype(np.int64)
            fraction = pos-index.astype(np.float32)
            # OpenCV clips horizontal coefficients but only clips row indices.
            if horizontal:
                fraction = np.where((index < 0) | (index >= source-1), 0, fraction)
            self.axes[key] = tuple(torch.as_tensor(v, device=self.device) for v in (
                np.clip(index, 0, source-1), np.clip(index+1, 0, source-1),
                np.rint((1-fraction)*2048).astype(np.int32),
                np.rint(fraction*2048).astype(np.int32)))
        return self.axes[key]

    def resize(self, rgb):
        if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError('Device RGB requires uint8 BHWC RGB')
        if rgb.device != self.device:
            raise ValueError('RGB must already be on the selected device')
        x0, x1, a0, a1 = self._axis(rgb.shape[2], True)
        y0, y1, b0, b1 = self._axis(rgb.shape[1], False)
        horizontal = (rgb[:, :, x0].to(torch.int32)*a0[None, None, :, None]
                      + rgb[:, :, x1].to(torch.int32)*a1[None, None, :, None])
        upper = (horizontal[:, y0] >> 4)*b0[None, :, None, None] >> 16
        lower = (horizontal[:, y1] >> 4)*b1[None, :, None, None] >> 16
        return ((upper+lower+2) >> 2).to(torch.uint8)

    def __call__(self, rgb):
        resized = self.resize(rgb)
        return torch.stack([self.lut[c][resized[..., c].long()] for c in range(3)], dim=1)

    @torch.no_grad()
    def verify(self, rgb):
        """Fail closed if the installed reference and device path differ at all."""
        import cv2
        sample = rgb[:2]
        reference_rgb = np.stack([cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                                  for im in sample.cpu().numpy()])
        actual_rgb = self.resize(sample).cpu().numpy()
        if not np.array_equal(actual_rgb, reference_rgb):
            delta = np.abs(actual_rgb.astype(int)-reference_rgb.astype(int))
            raise RuntimeError(f'Device RGB differs from OpenCV (max={delta.max()}); use --rgb-backend opencv')
        reference = self.processor(images=list(reference_rgb), return_tensors='pt', do_resize=False,
                                   do_center_crop=False)['pixel_values']
        actual = self(sample).cpu()
        if not torch.equal(actual, reference):
            raise RuntimeError('Device RGB normalization differs from processor; use --rgb-backend opencv')
        self.verified_real = True
        return dict(frames=len(sample), pixel_max_error=0, normalized_max_error=0)
