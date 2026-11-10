"""NFPLight heads with UNet++ dense nested skip pathways.

Topology reference: MrGiovanni/UNetPlusPlus, official PyTorch implementation,
pytorch/nnunet/network_architecture/generic_UNetPlusPlus.py.
NAF blocks, learned down/up sampling and material heads follow this repository.
Only the deepest full-resolution output is supervised.
"""

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from nfplight.network import NAFBlock


class NestedDecoder(Module):
    def __init__(self, dim, depths, *, dropout, norm_eps):
        super().__init__()
        self.stages = len(depths)
        self.ups = nn.ModuleDict()
        self.fusions = nn.ModuleDict()
        for column in range(1, self.stages + 1):
            for level in range(self.stages - column + 1):
                channels = dim * 2 ** level
                key = f'{level}_{column}'
                self.ups[key] = nn.Sequential(
                    nn.Conv2d(channels * 2, channels * 4, 1, bias = False),
                    nn.PixelShuffle(2))
                self.fusions[key] = nn.Sequential(
                    nn.Conv2d(channels * (column + 1), channels, 1),
                    *(NAFBlock(channels, dropout = dropout, norm_eps = norm_eps)
                      for _ in range(depths[self.stages - level - 1])))

    def forward(self, encoded):
        nodes = [[value] for value in encoded]
        for column in range(1, self.stages + 1):
            for level in range(self.stages - column + 1):
                key = f'{level}_{column}'
                deeper = self.ups[key](nodes[level + 1][column - 1])
                fused = torch.cat((*nodes[level], deeper), dim = 1)
                nodes[level].append(self.fusions[key](fused))
        return nodes[0][-1]


class NestedModel(Module):
    def __init__(self, *, dim, encoder_depths, middle_depth, decoder_depths,
                 appearance_channels, dropout, norm_eps):
        super().__init__()
        depths = [*encoder_depths, middle_depth, *decoder_depths]
        if (type(dim) is not int or dim < 1 or not encoder_depths
                or len(encoder_depths) != len(decoder_depths)
                or any(type(depth) is not int or depth < 0 for depth in depths)
                or appearance_channels not in (18, 30)
                or not 0 <= dropout < 1 or not 0 < norm_eps < 1):
            raise ValueError('invalid network architecture')
        self.appearance_channels = appearance_channels
        self.padder_size = 2 ** len(encoder_depths)
        self.intro_albedo = nn.Conv2d(appearance_channels, dim, 3, padding = 1)
        self.intro_specular = nn.Conv2d(3, dim, 3, padding = 1)
        self.albedoMap = nn.Sequential(nn.Conv2d(dim, 6, 3, padding = 1), nn.Tanh())
        self.specularMap = nn.Sequential(nn.Conv2d(dim * 2, 4, 3, padding = 1), nn.Tanh())
        for branch in ('albedo', 'specular'):
            encoders, downs = ModuleList(), ModuleList()
            channels = dim
            for depth in encoder_depths:
                encoders.append(nn.Sequential(
                    *(NAFBlock(channels, dropout = dropout, norm_eps = norm_eps)
                      for _ in range(depth))))
                downs.append(nn.Conv2d(channels, channels * 2, 2, 2))
                channels *= 2
            setattr(self, f'encoders_{branch}', encoders)
            setattr(self, f'downs_{branch}', downs)
            setattr(self, f'middle_blks_{branch}', nn.Sequential(
                *(NAFBlock(channels, dropout = dropout, norm_eps = norm_eps)
                  for _ in range(middle_depth))))
        self.decoder_albedo = NestedDecoder(dim, decoder_depths,
                                            dropout = dropout, norm_eps = norm_eps)
        self.decoder_specular = NestedDecoder(dim * 2, decoder_depths,
                                              dropout = dropout, norm_eps = norm_eps)

    def encode(self, x, branch):
        x = getattr(self, f'intro_{branch}')(x)
        encoded = []
        for encoder, down in zip(getattr(self, f'encoders_{branch}'),
                                 getattr(self, f'downs_{branch}')):
            x = encoder(x)
            encoded.append(x)
            x = down(x)
        encoded.append(getattr(self, f'middle_blks_{branch}')(x))
        return encoded

    def forward(self, x):
        if x.ndim != 4 or min(x.shape) < 1 or x.shape[1] != self.appearance_channels + 3:
            raise ValueError('expected nonempty BCHW features with the configured channel order')
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError('network features must be finite FP32')
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % self.padder_size, 0, (-height) % self.padder_size))
        appearance, relation = x.split((self.appearance_channels, 3), dim = 1)
        albedo = self.encode(appearance, 'albedo')
        specular = self.encode(relation, 'specular')
        diffuse, reflectance = self.albedoMap(self.decoder_albedo(albedo)).split(3, dim = 1)
        combined = [torch.cat((a, s), dim = 1) for a, s in zip(albedo, specular)]
        normal, roughness = self.specularMap(self.decoder_specular(combined)).split((3, 1), dim = 1)
        return torch.cat((normal, diffuse, roughness, reflectance), dim = 1)[:, :, :height, :width]
