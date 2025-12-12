
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.base_net import LayerNorm2d


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class TwoBranchNet(nn.Module):
    def __init__(self, width=32, middle_blk_num=12, enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
        super().__init__()
        self.intro_albedo = nn.Conv2d(in_channels=18, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.intro_specular = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                        bias=True)
        self.albedoMap = nn.Sequential(
                            nn.Conv2d(in_channels=width, out_channels=6, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
                            nn.Tanh()
                            )
        self.specularMap = nn.Sequential(
                    nn.Conv2d(in_channels=width*2, out_channels=4, kernel_size=3, padding=1, stride=1, groups=1,
                        bias=True),
                    nn.Tanh()
                    )

        self.encoders_albedo = nn.ModuleList()
        self.encoders_specular = nn.ModuleList()
        self.decoders_albedo = nn.ModuleList()
        self.decoders_specular = nn.ModuleList()
        self.middle_blks_albedo = nn.ModuleList()
        self.middle_blks_specular = nn.ModuleList()
        self.ups_albedo = nn.ModuleList()
        self.ups_specular = nn.ModuleList()
        self.downs_albedo = nn.ModuleList()
        self.downs_specular = nn.ModuleList()

        ### encoder for albedo
        chan = width
        for num in enc_blk_nums:
            self.encoders_albedo.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs_albedo.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_albedo = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        chan_albedo = chan

        ### encoder for specular
        chan = width
        for num in enc_blk_nums:
            self.encoders_specular.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs_specular.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_specular = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        chan_specular = chan
        
        ### decoder for albedo
        chan = chan_albedo    
        for num in dec_blk_nums:
            self.ups_albedo.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders_albedo.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            
        ### decoder for specular
        chan = chan_specular + chan_albedo    
        for num in dec_blk_nums:
            self.ups_specular.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders_specular.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders_albedo)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        inp_albedo, inp_specular = torch.split(inp, [18, 3], dim=1)

        ### encoder for albedo
        x = self.intro_albedo(inp_albedo)
        encs_albedo = []
        for encoder, down in zip(self.encoders_albedo, self.downs_albedo):
            x = encoder(x)
            encs_albedo.append(x)
            x = down(x)
        feature_albedo = self.middle_blks_albedo(x)
        
        ### encoder for specular
        x = self.intro_specular(inp_specular)
        encs_specular = []
        for encoder, down in zip(self.encoders_specular, self.downs_specular):
            x = encoder(x)
            encs_specular.append(x)
            x = down(x)
        feature_specuar = self.middle_blks_specular(x)

        ### decoder for albedo
        x = feature_albedo
        for decoder, up, enc_skip in zip(self.decoders_albedo, self.ups_albedo, encs_albedo[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        albedo_map = self.albedoMap(x)

        ### decoder for specular
        x = torch.cat([feature_specuar, feature_albedo], dim=1)
        for decoder, up, enc_skip_alb,enc_skip_spl in zip(self.decoders_specular, self.ups_specular,encs_albedo[::-1], encs_specular[::-1]):
            x = up(x)
            enc_skip = torch.cat([enc_skip_alb, enc_skip_spl], dim=1)
            x = x + enc_skip
            x = decoder(x)
        specular_map = self.specularMap(x)
        d, s = torch.split(albedo_map,[3,3],1)
        n, r = torch.split(specular_map,[3,1],1)
        svbrdf = torch.cat([n,d,r,s],dim=1)
        return svbrdf[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class TwoBranchRealNet(nn.Module):

    def __init__(self, width=32, middle_blk_num=12, enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
        super().__init__()
        self.intro_albedo = nn.Conv2d(in_channels=30, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.intro_specular = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                        bias=True)
        self.albedoMap = nn.Sequential(
                            nn.Conv2d(in_channels=width, out_channels=6, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
                            nn.Tanh()
                            )
        self.specularMap = nn.Sequential(
                    nn.Conv2d(in_channels=width*2, out_channels=4, kernel_size=3, padding=1, stride=1, groups=1,
                        bias=True),
                    nn.Tanh()
                    )

        self.encoders_albedo = nn.ModuleList()
        self.encoders_specular = nn.ModuleList()
        self.decoders_albedo = nn.ModuleList()
        self.decoders_specular = nn.ModuleList()
        self.middle_blks_albedo = nn.ModuleList()
        self.middle_blks_specular = nn.ModuleList()
        self.ups_albedo = nn.ModuleList()
        self.ups_specular = nn.ModuleList()
        self.downs_albedo = nn.ModuleList()
        self.downs_specular = nn.ModuleList()

        ### encoder for albedo
        chan = width
        for num in enc_blk_nums:
            self.encoders_albedo.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs_albedo.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_albedo = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        chan_albedo = chan

        ### encoder for specular
        chan = width
        for num in enc_blk_nums:
            self.encoders_specular.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs_specular.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_specular = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        chan_specular = chan
        
        ### decoder for albedo
        chan = chan_albedo    
        for num in dec_blk_nums:
            self.ups_albedo.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders_albedo.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            
        ### decoder for specular
        chan = chan_specular + chan_albedo    
        for num in dec_blk_nums:
            self.ups_specular.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders_specular.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders_albedo)

    def forward(self, inp):
        B, C, H, W = inp.shape
        # inp = self.check_image_size(inp)
        inp_albedo, inp_specular = torch.split(inp, [30, 3], dim=1)

        ### encoder for albedo
        x = self.intro_albedo(inp_albedo)
        encs_albedo = []
        for encoder, down in zip(self.encoders_albedo, self.downs_albedo):
            x = encoder(x)
            encs_albedo.append(x)
            x = down(x)
        feature_albedo = self.middle_blks_albedo(x)
        
        ### encoder for specular
        x = self.intro_specular(inp_specular)
        encs_specular = []
        for encoder, down in zip(self.encoders_specular, self.downs_specular):
            x = encoder(x)
            encs_specular.append(x)
            x = down(x)
        feature_specuar = self.middle_blks_specular(x)

        ### decoder for albedo
        x = feature_albedo
        for decoder, up, enc_skip in zip(self.decoders_albedo, self.ups_albedo, encs_albedo[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        albedo_map = self.albedoMap(x)

        ### decoder for specular
        x = torch.cat([feature_specuar, feature_albedo], dim=1)
        for decoder, up, enc_skip_alb,enc_skip_spl in zip(self.decoders_specular, self.ups_specular,encs_albedo[::-1], encs_specular[::-1]):
            x = up(x)
            enc_skip = torch.cat([enc_skip_alb, enc_skip_spl], dim=1)
            x = x + enc_skip
            x = decoder(x)
        specular_map = self.specularMap(x)
        d, s = torch.split(albedo_map,[3,3],1)
        n, r = torch.split(specular_map,[3,1],1)
        svbrdf = torch.cat([n,d,r,s],dim=1)
        return svbrdf[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class DenoiseNet(nn.Module):

    def __init__(self, width=32, middle_blk_num=12, enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
        super().__init__()

    
        self.intro = nn.Conv2d(in_channels=12, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Sequential(
                            nn.Conv2d(in_channels=width, out_channels=6, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
                            nn.Softplus()
                            )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        return x


def torch_norm(arr, dim=1):
    length = torch.sqrt(torch.sum(arr * arr, dim = dim, keepdims=True))
    return arr / (length + 1e-12)