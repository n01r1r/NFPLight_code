from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nfplight.data import map_loss
from nfplight.pipeline import TrainingPipeline, validation_metrics


def test_resume_rejects_changed_validation_batch_without_changing_train_contract():
    from nfplight.cli import validate_resume
    from nfplight.training import batch_contract
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / 'configs/nested_32_noquant.json').read_text())
    runtime = json.loads((root / 'configs/runtime_accum4.json').read_text())
    legacy = {key:value for key, value in runtime.items() if key != 'validation_batch_size'}
    assert batch_contract(runtime) == batch_contract(legacy)
    with pytest.raises(ValueError, match = 'validation_batch_size'):
        validate_resume({'clean_profile':profile, 'clean_runtime':legacy}, {}, profile, runtime, device = 'cuda')


def test_metrics_keep_each_crop_mask_weight_and_fp32():
    target = torch.zeros(2, 10, 2, 3)
    target[:, 2] = 1
    pred = target.clone()
    pred[0, 3:] = .2
    pred[1, 3:] = .8
    mask = torch.ones(2, 1, 2, 3, dtype = torch.bool)
    mask[0] = False
    mask[0, 0, 0, 0] = True
    reference = torch.zeros(2, 3, 3, 2, 3)
    rendered = torch.ones_like(reference)
    rendered[1] *= 3
    values = validation_metrics(pred, target, mask, rendered, reference)
    expected = []
    for index in range(2):
        p, t, m = pred[index:index + 1], target[index:index + 1], mask[index:index + 1]
        _, parts = map_loss(p, t, m)
        delta = (rendered[index:index + 1] - reference[index:index + 1]).abs()
        render = delta.masked_select(m[:, None].expand_as(delta)).mean()
        angle = torch.rad2deg(torch.acos((p[:, :3] * t[:, :3]).sum(1, keepdim = True).clamp(-1, 1))).masked_select(m).mean()
        expected.append(torch.stack([*parts, render, angle]))
    assert values.dtype == torch.float32
    torch.testing.assert_close(values, torch.stack(expected), rtol = 1e-6, atol = 1e-6)
    assert values[:, 1].mean().item() == pytest.approx(.5)
    pooled = map_loss(pred, target, mask)[1][1]
    assert not torch.isclose(values[:, 1].mean(), pooled)
    with pytest.raises(ValueError):
        validation_metrics(pred, target, torch.zeros_like(mask), rendered, reference)
    with pytest.raises(ValueError):
        validation_metrics(pred.half(), target, mask, rendered, reference)
    rendered[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        validation_metrics(pred, target, mask, rendered, reference)


def test_oom_retry_tail_and_material_group_means_preserve_population(monkeypatch):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.validation_batch_size = 4
    pipeline.runtime = {'validation_crops':[0, 4, 8]}
    pipeline.validation_lights = [None] * 12
    records = [{'material_id':str(index), 'category':'Fabric' if index != 1 else 'Leather',
                'group_id':'shared' if index < 2 else 'other'} for index in range(3)]
    pipeline.val = SimpleNamespace(records = records,
        crop = lambda index, crop: torch.tensor([index * 100 + crop]))
    successful, attempts, emptied, progress = [], [], [], []
    def batch(network, targets):
        attempts.append(len(targets))
        if len(targets) > 2:
            raise torch.cuda.OutOfMemoryError('simulated validation OOM')
        ids = targets[:, 0].numpy().astype(np.float64)
        successful.extend(ids.tolist())
        return np.repeat(ids[:, None], 6, axis = 1)
    pipeline.validation_batch = batch
    pipeline.progress = lambda phase, done, total, **details: progress.append((done, total, details))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: emptied.append(True))
    heldout = pipeline.evaluate(object(), 'map', 1000)['heldout']
    assert attempts == [4, 2, 2, 2, 2, 1] and len(emptied) == 1
    expected = [index * 100 + crop for index in range(3) for crop in (0, 4, 8)]
    assert successful == expected
    assert heldout['mean']['map_normal'] == 104
    assert heldout['source_group_macro']['map_normal'] == 129
    assert [row['map_normal'] for row in heldout['materials']] == [4, 104, 204]
    assert heldout['validation_batch_sizes_used'] == [1, 2]
    assert heldout['validation_batch_size_requested'] == 4 and heldout['validation_batch_size_final'] == 2
    assert heldout['validation_oom_retries'] == 1 and progress[-1][:2] == (9, 9)
    attempts.clear()
    pipeline.evaluate(object(), 'render', 1000)
    assert max(attempts) == 2
    pipeline.validation_batch = lambda network, targets: (_ for _ in ()).throw(ValueError('invalid data'))
    with pytest.raises(ValueError, match = 'invalid data'):
        pipeline.evaluate(object(), 'map', 2000)
