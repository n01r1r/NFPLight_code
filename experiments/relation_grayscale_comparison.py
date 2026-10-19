"""User-authorized measured-inference ablation; training/model defaults stay intact.

Linear RGB weights (0.299, 0.587, 0.114), no gamma. Grayscale is
computed before absolute difference, unlike the author's mean(abs(RGB delta)).
"""
from contextlib import contextmanager
from types import MethodType
import torch
from model.relation_geometry import spatial_far


def relation_numerator(signed_rgb, method):
    if signed_rgb.ndim != 4 or signed_rgb.shape[1] != 3:
        raise ValueError('expected signed RGB [B,3,H,W]')
    if not signed_rgb.is_floating_point() or not torch.isfinite(signed_rgb).all():
        raise ValueError('signed RGB must be finite floating point')
    if method == 'rgb_mean':
        return signed_rgb.abs().mean(1, keepdim=True)
    if method == 'weighted_grayscale':
        weights = signed_rgb.new_tensor([.299, .587, .114]).reshape(1, 3, 1, 1)
        return (signed_rgb * weights).sum(1, keepdim=True).abs()
    raise ValueError(f'unknown relation aggregation: {method}')


def _weighted_relation(self, near, aligned_far, valid_mask=None):
    signed = spatial_far(aligned_far, self.time_co_map, self.spatial_attenuation_factor).clamp(0, 1) - near
    ratio = relation_numerator(signed, 'weighted_grayscale') / self.coefficient.clamp(min=1e-5)
    return (self.indenty - self.ClipToOne(ratio, valid_mask),
            self.indenty - self.ClipToOne(ratio.clamp(min=1e-5).log(), valid_mask))


@contextmanager
def weighted_grayscale_relation(model):
    """Temporary experimental override, restored even if inference fails."""
    had_override = 'relationmapExtraction' in model.__dict__
    previous = model.__dict__.get('relationmapExtraction')
    model.relationmapExtraction = MethodType(_weighted_relation, model)
    try:
        yield
    finally:
        if had_override:
            model.relationmapExtraction = previous
        else:
            del model.relationmapExtraction
