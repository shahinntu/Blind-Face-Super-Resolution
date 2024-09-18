import random
import math

import numpy as np
import torch
from torch.nn import functional as F
from basicsr.data.degradations import (
    circular_lowpass_kernel,
    random_mixed_kernels,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.data.transforms import paired_random_crop


class Blur:
    def __init__(
        self,
        sinc_prob,
        kernel_list,
        kernel_prob,
        blur_sigma,
        betag_range,
        betap_range,
    ):
        self._sinc_prob = sinc_prob
        self._kernel_list = kernel_list
        self._kernel_prob = kernel_prob
        self._blur_sigma = blur_sigma
        self._betag_range = betag_range
        self._betap_range = betap_range

        self._kernel_range = [2 * v + 1 for v in range(3, 11)]

    def __call__(self, img_tensor):
        kernels = [self._create_kernel() for _ in range(img_tensor.size(0))]
        kernels = torch.stack(kernels, dim=0).to(img_tensor.device)

        return filter2D(img_tensor, kernels)

    def _create_kernel(self):
        kernel_size = random.choice(self._kernel_range)
        if np.random.uniform() < self._sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self._kernel_list,
                self._kernel_prob,
                kernel_size,
                self._blur_sigma,
                self._blur_sigma,
                [-math.pi, math.pi],
                self._betag_range,
                self._betap_range,
                noise_range=None,
            )
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel = torch.FloatTensor(kernel)

        return kernel


class Sinc:
    def __init__(self):
        self._kernel_range = [2 * v + 1 for v in range(3, 11)]

    def __call__(self, img_tensor):
        sinc_kernels = [self._create_kernel() for _ in range(img_tensor.size(0))]
        sinc_kernels = torch.stack(sinc_kernels, dim=0).to(img_tensor.device)

        return filter2D(img_tensor, sinc_kernels)

    def _create_kernel(self):
        kernel_size = random.choice(self._kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)

        return sinc_kernel


class RandomResize:
    def __init__(self, resize_prob, resize_range, scale=None):
        self._resize_prob = resize_prob
        self._resize_range = resize_range
        self._scale = scale

    def __call__(self, img_tensor, original_size):
        resize_type = random.choices(["up", "down", "keep"], self._resize_prob)[0]
        if resize_type == "up":
            resize_scale = np.random.uniform(1, self._resize_range[1])
        elif resize_type == "down":
            resize_scale = np.random.uniform(self._resize_range[0], 1)
        else:
            resize_scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])

        if self._scale:
            return F.interpolate(
                img_tensor,
                size=(
                    int(original_size[0] / self._scale * resize_scale),
                    int(original_size[1] / self._scale * resize_scale),
                ),
                mode=mode,
            )

        return F.interpolate(img_tensor, scale_factor=resize_scale, mode=mode)


class AddRandomNoise:
    def __init__(
        self, gray_noise_prob, gaussian_noise_prob, noise_range, poisson_scale_range
    ):
        self._gray_noise_prob = gray_noise_prob
        self._gaussian_noise_prob = gaussian_noise_prob
        self._noise_range = noise_range
        self._poisson_scale_range = poisson_scale_range

    def __call__(self, img_tensor):
        if np.random.uniform() < self._gaussian_noise_prob:
            img_tensor = random_add_gaussian_noise_pt(
                img_tensor,
                sigma_range=self._noise_range,
                clip=True,
                rounds=False,
                gray_prob=self._gray_noise_prob,
            )
        else:
            img_tensor = random_add_poisson_noise_pt(
                img_tensor,
                scale_range=self._poisson_scale_range,
                clip=True,
                rounds=False,
                gray_prob=self._gray_noise_prob,
            )

        return img_tensor


class JPEGCompression:
    def __init__(self, jpeger, jpeg_range):
        self._jpeger = jpeger
        self._jpeg_range = jpeg_range

    def __call__(self, img_tensor):
        jpeg_p = img_tensor.new_zeros(img_tensor.size(0)).uniform_(*self._jpeg_range)
        img_tensor = torch.clamp(img_tensor, 0, 1)

        return self._jpeger(img_tensor, quality=jpeg_p)


class FinalFilter:
    def __init__(self, resize, jpeg_compression, sinc, final_sinc_prob):
        self._resize = resize
        self._jpeg_compression = jpeg_compression
        self._sinc = sinc
        self._final_sinc_prob = final_sinc_prob

    def __call__(self, img_tensor, original_size):
        if np.random.uniform() < 0.5:
            img_tensor = self._resize(img_tensor, original_size)
            if np.random.uniform() < self._final_sinc_prob:
                img_tensor = self._sinc(img_tensor)
            img_tensor = self._jpeg_compression(img_tensor)
        else:
            img_tensor = self._jpeg_compression(img_tensor)
            img_tensor = self._resize(img_tensor, original_size)
            if np.random.uniform() < self._final_sinc_prob:
                img_tensor = self._sinc(img_tensor)

        return img_tensor


class CombinedFilter:
    def __init__(self, blur_config, filter_config, scale, gt_size, device):
        self._scale = scale
        self._gt_size = gt_size

        self._jpeger = DiffJPEG(differentiable=False).to(device)
        self._usm_sharpener = USMSharp().to(device)

        self._blur = Blur(
            blur_config.SINC_PROB,
            blur_config.KERNEL_LIST,
            blur_config.KERNEL_PROB,
            blur_config.BLUR_SIGMA,
            blur_config.BETAG_RANGE,
            blur_config.BETAP_RANGE,
        )
        self._random_resize = RandomResize(
            filter_config.RESIZE_PROB, filter_config.RESIZE_RANGE
        )
        self._add_random_noise = AddRandomNoise(
            filter_config.GRAY_NOISE_PROB,
            filter_config.GAUSSIAN_NOISE_PROB,
            filter_config.NOISE_RANGE,
            filter_config.POISSON_SCALE_RANGE,
        )
        self._jpeg_compression = JPEGCompression(self._jpeger, filter_config.JPEG_RANGE)

        self._second_blur_prob = filter_config.SECOND_BLUR_PROB
        self._blur2 = Blur(
            blur_config.SINC_PROB2,
            blur_config.KERNEL_LIST2,
            blur_config.KERNEL_PROB2,
            blur_config.BLUR_SIGMA2,
            blur_config.BETAG_RANGE2,
            blur_config.BETAP_RANGE2,
        )
        self._random_resize2 = RandomResize(
            filter_config.RESIZE_PROB2, filter_config.RESIZE_RANGE2, scale
        )
        self._add_random_noise2 = AddRandomNoise(
            filter_config.GRAY_NOISE_PROB2,
            filter_config.GAUSSIAN_NOISE_PROB2,
            filter_config.NOISE_RANGE2,
            filter_config.POISSON_SCALE_RANGE2,
        )

        final_resize = RandomResize([0, 0, 1], [1, 1], scale)
        final_jpeg_compression = JPEGCompression(
            self._jpeger, filter_config.JPEG_RANGE2
        )
        final_sinc = Sinc()
        self._final_filter = FinalFilter(
            final_resize,
            final_jpeg_compression,
            final_sinc,
            filter_config.FINAL_SINC_PROB,
        )

    @torch.no_grad()
    def __call__(self, img_tensor):
        original_size = img_tensor.shape[2:]
        img_tensor_usm = self._usm_sharpener(img_tensor)

        out = self._blur(img_tensor_usm)
        out = self._random_resize(out, original_size)
        out = self._add_random_noise(out)
        out = self._jpeg_compression(out)

        if np.random.uniform() < self._second_blur_prob:
            out = self._blur2(out)
        out = self._random_resize2(out, original_size)
        out = self._add_random_noise2(out)

        out = self._final_filter(out, original_size)

        lq_tensor = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        (img_tensor, img_tensor_usm), lq_tensor = paired_random_crop(
            [img_tensor, img_tensor_usm], lq_tensor, self._gt_size, self._scale
        )

        return lq_tensor, img_tensor, img_tensor_usm
