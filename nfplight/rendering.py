import math

import torch

from einops import rearrange, repeat


def normalize(x, dim):
    return x / (torch.sqrt(torch.sum(x * x, dim = dim, keepdim = True)) + 1e-12)


class Renderer:
    def __init__(self, config, *, device):
        self.config = config
        self.device = torch.device(device)
        axis = torch.linspace(-config['extent'], config['extent'], config['size'], device = device, dtype = torch.float32)
        y, x = torch.meshgrid(axis, axis, indexing = 'ij')
        self.surface = torch.stack((x, -y, torch.zeros_like(x)), dim = 0)[None]

    def geometry(self, lights, views):
        if (lights.ndim != 2 or lights.shape != views.shape or lights.shape[1] != 3 or len(lights) == 0
                or not torch.isfinite(lights).all() or not torch.isfinite(views).all()
                or (lights[:, 2] <= 0).any() or (views[:, 2] <= 0).any()):
            raise ValueError('lights/views must be finite L3 above the surface in model units')
        lights = rearrange(lights, 'l c -> l c 1 1') - self.surface
        views = rearrange(views, 'l c -> l c 1 1') - self.surface
        return normalize(lights, 1), normalize(views, 1), lights.square().sum(1, keepdim = True)

    def shade(self, maps, light, view, distance):
        # Maps B10HW; directions B,L,3,H,W; output B,L,3,H,W linear radiance.
        n, d, r, s = [rearrange(maps[:, section], 'b c h w -> b 1 c h w') for section in
                      (slice(0, 3), slice(3, 6), slice(6, 7), slice(7, 10))]
        r, d, s = r * .5 + .5, d * .5 + .5, s * .5 + .5
        eps = torch.ones_like(distance) * self.config['epsilon']
        n = normalize(n, -3)
        half = normalize((light + view) * .5, -3)
        r = torch.max(r, torch.ones_like(r) * self.config['roughness_floor'])
        def dot(a, b):
            return torch.max(torch.sum(a * b, dim = -3, keepdim = True), eps)
        nh, nv, nl, vh = dot(n, half), dot(n, view), dot(n, light), dot(view, half)
        alpha = r * r
        tmp = alpha / torch.max(eps, nh * nh * (alpha * alpha - 1.) + 1.)
        distribution = tmp * tmp * (1. / math.pi)
        k = torch.max(eps, r * r * .5)
        geometry = (1 / (nl * (1. - k) + k)) * (1 / (nv * (1. - k) + k))
        fresnel = s + (1. - s) * (2 ** (vh * (-5.55473 * vh - 6.98316)))
        specular = distribution * geometry * fresnel * .25
        result = (d * (1. / math.pi) + specular) * nl / distance
        return result * self.config['lamp_intensity']

    def __call__(self, maps, lights, views):
        if (maps.ndim != 4 or maps.shape[1:] != (10, self.config['size'], self.config['size'])
                or len(maps) == 0 or maps.dtype != torch.float32 or not torch.isfinite(maps).all()):
            raise ValueError('renderer requires finite FP32 B10HW maps at configured size')
        light, view, distance = self.geometry(lights, views)
        batch, count = len(maps), len(lights)
        maps = repeat(maps, 'b c h w -> (b l) c h w', l = count)
        light, view, distance = [repeat(value, 'l c h w -> (b l) 1 c h w', b = batch)
                                 for value in (light, view, distance)]
        result = self.shade(maps, light, view, distance)
        return rearrange(result, '(b l) 1 c h w -> b l c h w', b = batch, l = count)

    def observations(self, maps, near, far):
        # Keep legacy near-batch then far-batch ordering, including floating-point kernels.
        lights = maps.new_tensor([[0., 0., near], [0., 0., far]])
        directions, _, distances = self.geometry(lights, lights)
        directions = repeat(directions, 'l c h w -> (l b) 1 c h w', b = len(maps))
        distances = repeat(distances, 'l c h w -> (l b) 1 c h w', b = len(maps))
        doubled = torch.cat((maps, maps), dim = 0)
        result = self.shade(doubled, directions, directions, distances)
        result = rearrange(result, '(l b) 1 c h w -> b (l c) h w', l = 2)
        if not torch.isfinite(result).all() or (result < 0).any():
            raise ValueError('invalid synthetic linear observations')
        return result.clamp(0, 1)


def lights_from_config(config, *, device):
    radius = config['radius']
    return torch.tensor([[radius * math.sin(math.radians(theta)) * math.cos(math.radians(phi)),
                          radius * math.sin(math.radians(theta)) * math.sin(math.radians(phi)),
                          radius * math.cos(math.radians(theta))]
                         for theta in config['theta_degrees'] for phi in config['phi_degrees']],
                        dtype = torch.float32, device = device)
