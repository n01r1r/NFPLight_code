import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import numpy as np

from nfplight.network import NFPLight
from nfplight.storage import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_inference_cli_and_optimized_python_reject_nonfinite(tmp_path):
    config = json.loads((ROOT / 'configs/7m_spatial.json').read_text())
    config['network'].update(dim = 4, encoder_depths = [1], middle_depth = 1, decoder_depths = [1])
    write_json(tmp_path / 'config.json', config)
    network = NFPLight(**config['network'])
    torch.save({'net_g': network.state_dict()}, tmp_path / 'weight.pt')
    payload = {'schema': 'nfplight.input.v1', 'capture_gain_mode': 'author_patch20',
               'pair': torch.full((1, 6, 256, 256), .02),
               'valid_mask': torch.ones(1, 1, 256, 256, dtype = torch.bool)}
    torch.save(payload, tmp_path / 'pair.pt')
    arguments = ['-m', 'nfplight', 'infer', '--config', str(tmp_path / 'config.json'),
        '--input', str(tmp_path / 'pair.pt'), '--checkpoint', str(tmp_path / 'weight.pt'),
        '--out', str(tmp_path / 'prediction.pt')]
    result = subprocess.run([sys.executable, *arguments], cwd = ROOT, capture_output = True, text = True)
    assert result.returncode == 0, result.stderr
    pred = torch.load(tmp_path / 'prediction.pt', weights_only = True)
    assert pred['prediction'].shape == (1, 10, 256, 256)
    implicit = arguments.copy()
    position = implicit.index('--config')
    del implicit[position:position + 2]
    implicit[implicit.index('--out') + 1] = str(tmp_path / 'unidentified.pt')
    result = subprocess.run([sys.executable, *implicit], cwd = ROOT, capture_output = True, text = True)
    assert result.returncode != 0 and 'explicit --config' in result.stderr

    payload['pair'][0, 0, 0, 0] = float('nan')
    torch.save(payload, tmp_path / 'invalid.pt')
    arguments[arguments.index('--input') + 1] = str(tmp_path / 'invalid.pt')
    arguments[arguments.index('--out') + 1] = str(tmp_path / 'invalid_prediction.pt')
    result = subprocess.run([sys.executable, '-O', *arguments], cwd = ROOT, capture_output = True, text = True)
    assert result.returncode != 0 and 'ValueError' in result.stderr
    assert not (tmp_path / 'invalid_prediction.pt').exists()


def test_training_cli_synthetic_fixture_one_update(tmp_path):
    config = json.loads((ROOT / 'configs/7m_spatial.json').read_text())
    config['network'].update(dim = 4, encoder_depths = [1], middle_depth = 1, decoder_depths = [1])
    write_json(tmp_path / 'config.json', config)
    runtime = json.loads((ROOT / 'configs/runtime.json').read_text())
    runtime.update(sample_count = 16, batch_size = 1, cpu_threads = 2, disk_reserve_bytes = 0)
    write_json(tmp_path / 'runtime.json', runtime)
    shape = (16, 256, 256, 3)
    normal = np.full(shape, 128, dtype = np.uint8)
    normal[..., 2] = 255
    rows = []
    for split in ('train', 'val'):
        path = tmp_path / f'{split}.npz'
        np.savez_compressed(path, normal = normal, diffuse = np.full(shape, 100, dtype = np.uint8),
            roughness = np.full(shape[:-1], 128, dtype = np.uint8), specular = np.full(shape, 10, dtype = np.uint8))
        rows.append({'split': split, 'shard_path': str(path), 'material_id': split,
                     'group_id': split, 'category': 'Fabric'})
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in rows), encoding = 'utf-8')
    result = subprocess.run([sys.executable, '-m', 'nfplight', 'train', '--device', 'cpu',
        '--config', str(tmp_path / 'config.json'), '--runtime', str(tmp_path / 'runtime.json'),
        '--manifest', str(manifest), '--out', str(tmp_path / 'smoke'), '--smoke-updates', '1'],
        cwd = ROOT, capture_output = True, text = True)
    assert result.returncode == 0, result.stderr
    payload = torch.load(tmp_path / 'smoke/last.pt', weights_only = True)
    assert payload['state']['total_step'] == 1 and payload['state']['status'] == 'interrupted'
    assert payload['execution_purpose'] == 'smoke' and payload['identity']['purpose'] == 'smoke'


@pytest.mark.parametrize('section, key, value', [
    ('network', 'norm_eps', 1e-4),
    ('features', 'saturation', .9),
    ('features', 'log_epsilon', .02),
])
def test_exact_resume_rejects_changed_numerical_profile(section, key, value):
    from nfplight.cli import validate_resume

    profile = json.loads((ROOT / 'configs/7m_spatial.json').read_text())
    runtime = json.loads((ROOT / 'configs/runtime.json').read_text())
    changed = copy.deepcopy(profile)
    changed[section][key] = value
    for payload in ({}, {'clean_profile': profile}):
        with pytest.raises(ValueError, match = 'exact resume profile mismatch'):
            validate_resume(payload, {}, changed, runtime, device = 'cpu')
