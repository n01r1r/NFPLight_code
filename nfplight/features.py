import math

import torch

from einops import rearrange
import numpy as np


class Features:
    def __init__(self, config, *, device):
        for key in ('epsilon', 'log_epsilon', 'training_gain'):
            if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f'feature config requires finite positive {key}')
        if (config['gain_patch'] != [118, 138] or not 0 < config['saturation'] <= 1
                or config['geometry_dtype'] not in ('float32', 'float64')
                or config['normalization'] not in ('sample', 'batch')):
            raise ValueError('unsupported gain patch, saturation, geometry dtype or normalization')
        self.config = config
        self.device = torch.device(device)
        self.size = size = config['size']
        if size != 256 or config['channels'] not in (21, 33):
            raise ValueError('supported checkpoints require 256px and 21/33 channels')
        if config['relation'] not in ('legacy_no_spatial', 'spatial_inverse_square_v1'):
            raise ValueError('explicit known relation version is required')
        near, far = config['near'], config['far']
        if not (math.isfinite(near) and math.isfinite(far) and 0 < near < far):
            raise ValueError('require finite 0 < near < far in common model units')
        dtype = {'float32': np.float32, 'float64': np.float64}[config['geometry_dtype']]
        y, x = np.ogrid[:size, :size]
        x, y = x.astype(dtype), y.astype(dtype)
        radius = np.sqrt((x - dtype((size - 1) / 2)) ** 2 + (y - dtype((size - 1) / 2)) ** 2,
                         dtype = dtype) / dtype(size / 2)
        a, b = np.arctan(radius / near), np.arctan(radius / far)
        coefficient = (np.cos(b) - np.cos(a)) / np.cos(a)
        self.coefficient = torch.as_tensor(coefficient / coefficient.max(), dtype = torch.float32, device = device)[None, None]
        self.angular = torch.as_tensor(np.cos(a) / np.cos(b), dtype = torch.float32, device = device)[None, None]
        axis = (torch.arange(size, dtype = torch.float32, device = device) - (size - 1) / 2) / (size / 2.)
        radius_squared = axis[:, None].square() + axis[None, :].square()
        ratio = (radius_squared + far ** 2) / (radius_squared + near ** 2)
        center = (size - 1) // 2
        factor = ratio / ratio[center, center]
        self.spatial = factor[None, None] if config['relation'] == 'spatial_inverse_square_v1' else torch.ones_like(self.coefficient)
        self.device = self.coefficient.device

    def normalize(self, x, mask):
        if self.config['normalization'] == 'batch':
            low, high = x.amin(), x.amax()
        elif self.config['normalization'] == 'sample':
            low = (x if mask is None else x.masked_fill(~mask, float('inf'))).amin((1, 2, 3), keepdim = True)
            high = (x if mask is None else x.masked_fill(~mask, -float('inf'))).amax((1, 2, 3), keepdim = True)
        else:
            raise ValueError('unknown relation normalization')
        return (x - low) / (high - low).clamp(min = self.config['epsilon'])

    def __call__(self, pair, *, purpose, mask = None):
        if (pair.ndim != 4 or len(pair) < 1 or tuple(pair.shape[1:]) != (6, self.size, self.size)
                or pair.dtype != torch.float32 or pair.device != self.device
                or not torch.isfinite(pair).all() or pair.amin() < 0 or pair.amax() > 1):
            raise ValueError('pair must be finite FP32 [B,6,256,256] linear RGB in [0,1] on the feature device')
        if purpose not in ('capture', 'synthetic'):
            raise ValueError('purpose must be capture or synthetic')
        if self.config['channels'] == 33 and (len(pair) != 1 or purpose != 'capture'):
            raise ValueError('author real33 requires one measured pair')
        if mask is not None:
            if (mask.dtype != torch.bool or mask.shape != (len(pair), 1, self.size, self.size)
                    or mask.device != pair.device or not mask.flatten(1).any(1).all()):
                raise ValueError('mask must have nonempty boolean B1HW support on the input device')
            if self.config['channels'] == 33 and not mask.all():
                raise ValueError('author real33 requires full valid support')
            pair = pair * mask
        near, far = pair.chunk(2, dim = 1)
        eps = self.config['epsilon']
        if purpose == 'capture':
            start, end = self.config['gain_patch']
            if mask is not None and not mask[:, :, start:end, start:end].all():
                raise ValueError('capture gain patch requires full valid support')
            near_mean = near[:, :, start:end, start:end].mean((1, 2, 3))
            far_mean = far[:, :, start:end, start:end].mean((1, 2, 3))
            gain = near_mean / far_mean.clamp(min = eps)
        else:
            gain = near.new_full((len(near),), self.config['training_gain'])
            near_mean = far_mean = None
        aligned = far * rearrange(gain, 'b -> b 1 1 1')
        difference = (self.angular * (aligned * self.spatial)).clamp(0, 1) - near
        raw = difference.abs().mean(dim = 1, keepdim = True) / self.coefficient.clamp(min = eps)
        relation = 1. - self.normalize(raw, mask)
        log_relation = 1. - self.normalize(raw.clamp(min = eps).log(), mask)
        aligned = aligned.clamp(0, 1)
        saturated = (near.amax(1, keepdim = True) > self.config['saturation']) | (aligned.amax(1, keepdim = True) > self.config['saturation'])
        appearance = torch.cat((pair, aligned), dim = 1)
        if self.config['channels'] == 33:
            appearance = torch.cat((pair, pair, aligned), dim = 1)
            saturated = saturated | (far.amax(1, keepdim = True) > self.config['saturation'])
        log_eps = self.config['log_epsilon']
        if self.config['log_form'] == 'tensor':
            one = torch.ones((1,), dtype = pair.dtype, device = pair.device)
            epsilon = one * log_eps
            logged = ((appearance + log_eps).log() - epsilon.log()) / ((one + epsilon).log() - epsilon.log())
        elif self.config['log_form'] == 'scalar':
            logged = ((appearance + log_eps).log() - math.log(log_eps)) / (math.log1p(log_eps) - math.log(log_eps))
        else:
            raise ValueError('unknown logarithm contract')
        features = torch.cat((appearance, logged, relation, log_relation, (~saturated).float()), dim = 1)
        if mask is not None:
            features = features * mask
        features = features * 2 - 1
        if not torch.isfinite(features).all():
            raise ValueError('nonfinite features')
        return features, {'gain': gain, 'near_patch_mean': near_mean, 'far_patch_mean': far_mean,
                          'gain_floor_active': None if far_mean is None else far_mean < eps,
                          'relation': relation, 'log_relation': log_relation, 'aligned_far': aligned}
