"""Frozen bottleneck operators used only by the paired probe."""
import torch
from torch import nn

from network.base_net import LayerNorm2d
from network.nfplight_net import TwoBranchNet


class BottleneckMixer(nn.Module):
    """Apply one shared spatial or positionwise operator to a branch bottleneck."""

    def __init__(self, operator, bottleneck):
        super().__init__()
        if operator not in ("per_channel_shared_spatial_fc", "per_pixel_channel_mlp"):
            raise ValueError(f"unsupported bottleneck operator: {operator}")
        self.operator = operator
        self.channels = bottleneck["channels"]
        self.side = bottleneck["side"]
        self.norm = LayerNorm2d(self.channels)
        if operator == "per_channel_shared_spatial_fc":
            size, hidden = self.side * self.side, bottleneck["spatial_fc_hidden"]
        else:
            size, hidden = self.channels, bottleneck["pixel_mlp_hidden"]
        self.fc = nn.Sequential(
            nn.Linear(size, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, size, bias=False),
        )
        self.gate = nn.Parameter(torch.tensor(float(bottleneck["residual_gate_initial"])))
        self.enabled = True
        self.last_relative_residual = 0.0

    def forward(self, x):
        if not self.enabled:
            return x
        expected = (self.channels, self.side, self.side)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"probe requires [{expected[0]},{expected[1]},{expected[2]}] bottlenecks")
        z = self.norm(x)
        if self.operator == "per_channel_shared_spatial_fc":
            # Linear acts on each channel's flattened spatial plane with shared weights.
            residual = self.fc(z.flatten(2)).reshape_as(x)
        else:
            # Linear acts on channels independently at every spatial location.
            residual = self.fc(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        numerator = (self.gate * residual).detach().square().mean().sqrt()
        denominator = x.detach().square().mean().sqrt().clamp_min(1e-8)
        self.last_relative_residual = numerator / denominator
        return x + self.gate * residual


def create_network(arm_id, model_config):
    arms = {arm["id"]: arm for arm in model_config["arms"]}
    if arm_id not in arms:
        raise ValueError(f"unknown arm id: {arm_id}")
    backbone = model_config["backbone"]
    net = TwoBranchNet(
        width=backbone["width"],
        middle_blk_num=backbone["middle_blocks"],
        enc_blk_nums=backbone["encoder_blocks"],
        dec_blk_nums=backbone["decoder_blocks"],
    )
    operator = arms[arm_id]["operator"]
    if operator != "identity":
        for branch in (net.middle_blks_albedo, net.middle_blks_specular):
            branch.add_module("mixer", BottleneckMixer(operator, model_config["bottleneck"]))
    return net


def mixers(net):
    return [module for module in net.modules() if isinstance(module, BottleneckMixer)]


def parameter_count(net):
    return sum(parameter.numel() for parameter in net.parameters())
