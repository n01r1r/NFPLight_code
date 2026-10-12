import copy
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.robustness_suite.data import (
    AUGMENTATION_SETTINGS, additional_training_slots, audit_inputs,
    augment_material, overlap_pixels,
)


def target(batch=2):
    value = torch.zeros(batch, 10, 3, 4, dtype=torch.float32)
    value[:, 2] = 1
    value[:, 3:6] = torch.tensor([-0.6, 0.0, 0.6])[None, :, None, None]
    value[:, 6] = -0.2
    value[:, 7:10] = -0.92
    return value


def test_physical_mapping_shared_rgb_and_normals_unchanged():
    original = target()
    before = original.clone()
    result, metadata = augment_material(original, random.Random(30019))
    for i, draw in enumerate(metadata['items']):
        expected_diffuse = ((original[i, 3:6] + 1) / 2 * draw['diffuse_scale']).clamp(0, 1)
        expected_rough = ((original[i, 6:7] + 1) / 2 + draw['roughness_offset']).clamp(0, 1)
        assert torch.allclose((result[i, 3:6] + 1) / 2, expected_diffuse)
        assert torch.allclose((result[i, 6:7] + 1) / 2, expected_rough)
    assert torch.equal(original, before)
    assert torch.equal(result[:, :3], original[:, :3])
    assert torch.equal(result[:, 7:], original[:, 7:])
    assert result.dtype == torch.float32 and torch.isfinite(result).all()
    assert metadata['settings'] == AUGMENTATION_SETTINGS


def test_resume_rng_consumption_and_no_global_rng_pollution():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    rng = random.Random(30019)
    state = rng.getstate()
    first, first_metadata = augment_material(target(), rng)
    after = rng.getstate()
    replay = random.Random()
    replay.setstate(state)
    second, second_metadata = augment_material(target(), replay)
    assert torch.equal(first, second) and first_metadata == second_metadata
    control = random.Random()
    control.setstate(state)
    for _ in range(2):
        control.uniform(0.8, 1.2)
        control.uniform(-0.05, 0.05)
    assert control.getstate() == after == replay.getstate()
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    # Invalid input is rejected before consuming the dedicated stream.
    invalid = target()
    invalid[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        augment_material(invalid, rng)
    assert rng.getstate() == after


@pytest.mark.parametrize('change', ['shape', 'empty', 'double', 'nan', 'inf', 'range', 'normal'])
def test_invalid_target_rejected(change):
    value = target()
    if change == 'shape': value = value[:, :9]
    if change == 'empty': value = value[:0]
    if change == 'double': value = value.double()
    if change == 'nan': value[0, 3, 0, 0] = float('nan')
    if change == 'inf': value[0, 3, 0, 0] = float('inf')
    if change == 'range': value[0, 3, 0, 0] = 1.01
    if change == 'normal': value[:, :3] = 0
    with pytest.raises(ValueError):
        augment_material(value, random.Random(19))


def test_physical_clamping_and_d4_commutation():
    value = target(1)
    value[:, 3] = 1
    value[:, 4] = -1
    value[:, 6] = 1
    result, _ = augment_material(value, random.Random(2))
    assert result.min() >= -1 and result.max() <= 1
    assert (result[:, 3] == 1).all() and (result[:, 4] == -1).all()
    rotated = torch.rot90(value, 1, (-2, -1))
    rotated[:, 0], rotated[:, 1] = -rotated[:, 1].clone(), rotated[:, 0].clone()
    other, _ = augment_material(rotated, random.Random(2))
    expected = torch.rot90(result, 1, (-2, -1))
    expected[:, 0], expected[:, 1] = -expected[:, 1].clone(), expected[:, 0].clone()
    assert torch.equal(other, expected)


@pytest.mark.parametrize('first,second,edge,expected', [
    ((0, 0), (4, 0), 4, 0), ((0, 0), (0, 4), 4, 0),
    ((0, 0), (3, 3), 4, 1), ((0, 0), (0, 0), 4, 16),
    ((5, 7), (7, 9), 4, 4), ((0, 0), (8, 1), 4, 0),
])
def test_coordinate_overlap(first, second, edge, expected):
    assert overlap_pixels(first, second, edge) == expected
    assert overlap_pixels(second, first, edge) == expected


def execution_fixture(tmp_path, *, missing=False, second_material=False):
    records, train, unseen = [], [], []
    ids = ['materialA', 'materialB'] if second_material else ['materialA']
    for index, material in enumerate(ids):
        source = tmp_path / 'source' / 'train' / material
        source.mkdir(parents=True)
        shard = tmp_path / f'{material}.npz'
        origins = np.array([(0, 0), (0, 4), (0, 8), (4, 0), (4, 4), (4, 8), (8, 0), (8, 4), (8, 8)], np.int32)
        payload = {'resolution': np.array(4), 'origins': origins}
        rng = np.random.default_rng(19 + index)
        for key in ('normal', 'diffuse', 'roughness', 'specular'):
            shape = (12, 12) if key == 'roughness' else (12, 12, 3)
            native = rng.integers(0, 256, shape, np.uint8)
            Image.fromarray(native).save(source / f'{key}.png')
            payload[key] = np.stack([native[y:y+4, x:x+4] for y, x in origins])
        if missing and index == len(ids) - 1: payload.pop('origins')
        np.savez(shard, **payload)
        records.append({'material_id': material, 'official_split': 'train', 'split': 'train',
                        'group_id': material, 'shard_path': str(shard), 'source_dir': str(source)})
        base = {'material_id': material, 'category': 'Fabric', 'shard_path': str(shard)}
        train.extend([dict(base, crop=0), dict(base, crop=8)])
        unseen.append(dict(base, crop=4))
    records.append({'material_id': 'heldout', 'official_split': 'train', 'split': 'val', 'group_id': 'heldout'})
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in records), encoding='utf-8')
    return {'resolved_inputs': {'manifest': str(manifest), 'train': train, 'unseen_crop': unseen,
                               'heldout': [{'material_id': 'heldout', 'crop': 4}]},
            'scientific_settings': {'dataset': {'crops_per_shard': 9}, 'backbone': {'input_resolution': 4}}}


def test_audit_balanced_verified_extras_and_evidence_reload(tmp_path):
    execution = execution_fixture(tmp_path, second_material=True)
    audit = audit_inputs(execution)
    assert audit['eligible_extra_crop']
    assert len(audit['safe_train_slots']) == 6
    assert [slot['crop'] for slot in audit['selected_balanced_extras']] == [1, 1]
    assert all(row['verified_crops'] == list(range(9)) for row in audit['materials'])
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps(audit), encoding='utf-8')
    assert additional_training_slots(execution, path) == audit['safe_train_slots']
    Path(execution['resolved_inputs']['manifest']).write_text('changed')
    with pytest.raises(ValueError, match='evidence changed'):
        additional_training_slots(execution, path)


def test_missing_provenance_blocks_whole_balanced_arm(tmp_path):
    execution = execution_fixture(tmp_path, missing=True, second_material=True)
    audit = audit_inputs(execution)
    assert not audit['eligible_extra_crop']
    assert audit['safe_train_slots'] == execution['resolved_inputs']['train']
    assert audit['selected_balanced_extras'] == []
    assert 'missing crop origins' in audit['blocking_reasons'][0]


def test_false_coordinates_do_not_become_safe(tmp_path):
    execution = execution_fixture(tmp_path)
    shard = Path(execution['resolved_inputs']['train'][0]['shard_path'])
    with np.load(shard) as archive: payload = {key: archive[key] for key in archive.files}
    payload['origins'][1] = [4, 0]
    np.savez(shard, **payload)
    audit = audit_inputs(execution)
    assert not audit['eligible_extra_crop']
    assert 'differs from stated source' in audit['blocking_reasons'][0]


def test_overlap_is_recorded_and_candidate_rejected(tmp_path):
    execution = execution_fixture(tmp_path)
    shard = Path(execution['resolved_inputs']['train'][0]['shard_path'])
    with np.load(shard) as archive: payload = {key: archive[key] for key in archive.files}
    source = tmp_path / 'source' / 'train' / 'materialA'
    for crop in (1, 8):
        payload['origins'][crop] = payload['origins'][4]
        for key in ('normal', 'diffuse', 'roughness', 'specular'):
            with Image.open(source / f'{key}.png') as image: native = np.asarray(image)
            payload[key][crop] = native[4:8, 4:8]
    np.savez(shard, **payload)
    audit = audit_inputs(execution)
    row = audit['materials'][0]
    assert not row['candidates'][0]['verified_safe']
    assert row['selected_extra_crop'] == 2
    assert audit['baseline_train_unseen_overlap'][0]['overlap_pixels'] == 16


def test_official_test_paths_rejected_before_open(tmp_path):
    execution = execution_fixture(tmp_path)
    execution['resolved_inputs']['train'][0]['shard_path'] = str(tmp_path / 'test' / 'forbidden.npz')
    with pytest.raises(ValueError, match='official test path'):
        audit_inputs(execution)


def test_xy_axis_order_proven_by_exact_matching(tmp_path):
    execution = execution_fixture(tmp_path)
    shard = Path(execution['resolved_inputs']['train'][0]['shard_path'])
    with np.load(shard) as archive: payload = {key: archive[key] for key in archive.files}
    payload['origins'] = payload['origins'][:, ::-1]
    np.savez(shard, **payload)
    audit = audit_inputs(execution)
    assert audit['eligible_extra_crop']
    row = audit['materials'][0]
    assert row['stored_axis_order_verified'] == 'xy'
    assert row['origins_yx'][1] == [0, 4]
    assert row['stored_origins'][1] == [4, 0]


def test_missing_heldout_group_provenance_blocks_extras(tmp_path):
    execution = execution_fixture(tmp_path)
    manifest = Path(execution['resolved_inputs']['manifest'])
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[-1].pop('group_id')
    manifest.write_text('\n'.join(json.dumps(row) for row in records))
    audit = audit_inputs(execution)
    assert not audit['eligible_extra_crop']
    assert 'heldout source-group manifest provenance missing' in audit['blocking_reasons']
