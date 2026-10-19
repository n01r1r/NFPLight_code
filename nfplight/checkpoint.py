import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nfplight.network import NFPLight


def read_config(path):
    value = json.loads(Path(path).read_text(encoding = 'utf-8-sig'))
    for field in ('network', 'features', 'decoder', 'feature_version', 'profile'):
        if field not in value:
            raise ValueError(f'missing profile field: {field}')
    if value['decoder'] not in ('unit_xyz', 'raw'):
        raise ValueError('unsupported normal decoder')
    if value['features']['channels'] != value['network']['appearance_channels'] + 3:
        raise ValueError('feature channels disagree with network')
    return value


def load_payload(path):
    value = torch.load(Path(path), map_location = 'cpu', weights_only = True)
    if not isinstance(value, dict):
        raise ValueError('checkpoint must be a dictionary')
    return value


def state_dict(payload):
    containers = [key for key in ('net_g', 'params', 'params_ema', 'state_dict') if key in payload]
    if len(containers) > 1:
        raise ValueError('ambiguous checkpoint weight containers')
    values = payload[containers[0]] if containers else payload
    if not isinstance(values, dict) or not values:
        raise ValueError('empty network state')
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError('network state must map names to tensors')
        parts = name.split('.')
        while parts and parts[0] in ('module', '_orig_mod'):
            parts.pop(0)
        name = '.'.join(parts)
        if not name or name in result:
            raise ValueError('duplicate or empty normalized checkpoint key')
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'weight must be finite FP32: {name}')
        result[name] = value
    return result


def check_profile(payload, config):
    scientific = payload.get('identity', {}).get('scientific_settings', {})
    rendering = scientific.get('rendering', {})
    for actual, expected, label in (
        (payload.get('feature_version', rendering.get('feature_version')), config['feature_version'], 'feature version'),
        (payload.get('relation_version', rendering.get('relation_version')), config['features']['relation'], 'relation version'),
    ):
        if actual is not None and actual != expected:
            raise ValueError(f'checkpoint/config {label} conflict: {actual} != {expected}')
    backbone = payload.get('backbone', scientific.get('backbone'))
    if backbone is not None:
        expected = config['network']
        for old, new in (('width', 'dim'), ('encoder_blocks', 'encoder_depths'),
                         ('middle_blocks', 'middle_depth'), ('decoder_blocks', 'decoder_depths')):
            if backbone[old] != expected[new]:
                raise ValueError(f'checkpoint/config backbone conflict: {old}')
    decoder = payload.get('normal_decoder')
    if scientific.get('loss', {}).get('normal') == 'unit XYZ L1':
        decoder = 'unit XYZ'
    if decoder in ('unit XYZ', 'unit normalization before loss/inference') and config['decoder'] != 'unit_xyz':
        raise ValueError('checkpoint requires unit normal decoding')


def load_model(path, config, *, device):
    payload = load_payload(path)
    check_profile(payload, config)
    network = NFPLight(**config['network'])
    network.load_state_dict(state_dict(payload), strict = True)
    return network.to(device = device, dtype = torch.float32).eval()


def decode(raw, mode):
    if raw.ndim != 4 or raw.shape[1] != 10 or not torch.isfinite(raw).all():
        raise ValueError('prediction must be finite B10HW N/D/R/S')
    if mode == 'raw':
        return raw
    if mode != 'unit_xyz' or (raw[:, :3].norm(dim = 1) < 1e-8).any():
        raise ValueError('invalid decoder or near-zero predicted normal')
    return torch.cat((F.normalize(raw[:, :3], dim = 1), raw[:, 3:]), dim = 1)
