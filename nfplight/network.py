import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        mean = x.mean(1, keepdim = True)
        variance = (x - mean).pow(2).mean(1, keepdim = True)
        normalized = (x - mean) / (variance + eps).sqrt()
        ctx.save_for_backward(normalized, variance, weight)
        ctx.eps = eps
        return (rearrange(weight, 'c -> 1 c 1 1') * normalized
                + rearrange(bias, 'c -> 1 c 1 1'))

    @staticmethod
    def backward(ctx, gradient):
        normalized, variance, weight = ctx.saved_tensors
        scaled = gradient * rearrange(weight, 'c -> 1 c 1 1')
        mean = scaled.mean(dim = 1, keepdim = True)
        correlation = (scaled * normalized).mean(dim = 1, keepdim = True)
        # Preserve reduction and multiplication order for legacy Adam continuation.
        dx = 1. / torch.sqrt(variance + ctx.eps) * (scaled - normalized * correlation - mean)
        dw = (gradient * normalized).sum(dim = 3).sum(dim = 2).sum(dim = 0)
        db = gradient.sum(dim = 3).sum(dim = 2).sum(dim = 0)
        return dx, dw, db, None


class LayerNorm(Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Gate(Module):
    def forward(self, x):
        left, right = x.chunk(2, dim = 1)
        return left * right


class NAFBlock(Module):
    def __init__(self, dim, *, dropout, norm_eps):
        super().__init__()
        # Parameter registration order is part of the legacy optimizer contract.
        self.conv1 = nn.Conv2d(dim, dim * 2, 1)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 3, padding = 1, groups = dim * 2)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1))
        self.sg = Gate()
        self.conv4 = nn.Conv2d(dim, dim * 2, 1)
        self.conv5 = nn.Conv2d(dim, dim, 1)
        self.norm1 = LayerNorm(dim, norm_eps)
        self.norm2 = LayerNorm(dim, norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        residual = x
        x = self.sg(self.conv2(self.conv1(self.norm1(x))))
        x = self.conv3(x * self.sca(x))
        x = residual + self.dropout1(x) * self.beta
        residual = x
        x = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return residual + self.dropout2(x) * self.gamma


class NFPLight(Module):
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
        # Keep serialized names and traversal order; changing these breaks old Adam ids.
        self.intro_albedo = nn.Conv2d(appearance_channels, dim, 3, padding = 1)
        self.intro_specular = nn.Conv2d(3, dim, 3, padding = 1)
        self.albedoMap = nn.Sequential(nn.Conv2d(dim, 6, 3, padding = 1), nn.Tanh())
        self.specularMap = nn.Sequential(nn.Conv2d(dim * 2, 4, 3, padding = 1), nn.Tanh())
        for name in ('encoders_albedo', 'encoders_specular', 'decoders_albedo',
                     'decoders_specular', 'middle_blks_albedo', 'middle_blks_specular',
                     'ups_albedo', 'ups_specular', 'downs_albedo', 'downs_specular'):
            setattr(self, name, ModuleList())

        def blocks(channels, depth):
            return nn.Sequential(*(NAFBlock(channels, dropout = dropout, norm_eps = norm_eps)
                                   for _ in range(depth)))

        for branch in ('albedo', 'specular'):
            channels = dim
            for depth in encoder_depths:
                getattr(self, f'encoders_{branch}').append(blocks(channels, depth))
                getattr(self, f'downs_{branch}').append(nn.Conv2d(channels, channels * 2, 2, 2))
                channels *= 2
            setattr(self, f'middle_blks_{branch}', blocks(channels, middle_depth))
        bottleneck_dim = channels
        for branch, channels in (('albedo', bottleneck_dim), ('specular', bottleneck_dim * 2)):
            for depth in decoder_depths:
                getattr(self, f'ups_{branch}').append(nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias = False), nn.PixelShuffle(2)))
                channels //= 2
                getattr(self, f'decoders_{branch}').append(blocks(channels, depth))

    def encode(self, x, branch):
        x = getattr(self, f'intro_{branch}')(x)
        skips = []
        for encoder, down in zip(getattr(self, f'encoders_{branch}'), getattr(self, f'downs_{branch}')):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        return getattr(self, f'middle_blks_{branch}')(x), skips

    def decode(self, x, skips, branch):
        for decoder, up, skip in zip(getattr(self, f'decoders_{branch}'),
                                    getattr(self, f'ups_{branch}'), reversed(skips)):
            x = up(x) + skip
            x = decoder(x)
        return x

    def forward(self, x):
        # B,C,H,W: 18/30 appearance channels followed by relation/log-relation/mask.
        if x.ndim != 4 or min(x.shape) < 1 or x.shape[1] != self.appearance_channels + 3:
            raise ValueError('expected nonempty BCHW features with the configured channel order')
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError('network features must be finite FP32')
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % self.padder_size, 0, (-height) % self.padder_size))
        appearance, relation = x.split((self.appearance_channels, 3), dim = 1)
        albedo, albedo_skips = self.encode(appearance, 'albedo')
        specular, specular_skips = self.encode(relation, 'specular')
        diffuse, reflectance = self.albedoMap(self.decode(albedo, albedo_skips, 'albedo')).split(3, dim = 1)
        skips = [torch.cat((a, s), dim = 1) for a, s in zip(albedo_skips, specular_skips)]
        x = torch.cat((specular, albedo), dim = 1)
        normal, roughness = self.specularMap(self.decode(x, skips, 'specular')).split((3, 1), dim = 1)
        return torch.cat((normal, diffuse, roughness, reflectance), dim = 1)[:, :, :height, :width]
