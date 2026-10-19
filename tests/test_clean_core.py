import copy
import json
from pathlib import Path

import pytest
import torch

from nfplight.checkpoint import read_config, state_dict, decode
from nfplight.data import canonical_target, map_loss
from nfplight.features import Features
from nfplight.network import NFPLight
from nfplight.storage import atomic_write, output_lock
from nfplight.training import Trainer, evaluation_mode, capture_rng
from preprocess_dng import select_frames

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse = True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(7)


def profile():
    return read_config(ROOT / 'configs/7m_spatial.json')


def test_model_config_and_parameter_count():
    config = profile()['network']
    net = NFPLight(**config)
    assert sum(value.numel() for value in net.parameters()) == 7609834
    config.update(dim = 4, encoder_depths = [1], middle_depth = 1, decoder_depths = [1])
    small = NFPLight(**config)
    assert small(torch.zeros(1, 21, 9, 11)).shape == (1, 10, 9, 11)
    with pytest.raises(ValueError):
        small(torch.zeros(1, 33, 9, 11))


def test_capture_gain_is_not_training_gain_and_batch_is_independent():
    features = Features(profile()['features'], device = 'cpu')
    pair = torch.full((2, 6, 256, 256), .02)
    pair[0, :3] *= 2
    pair[1, :3] *= 3
    capture, trace = features(pair, purpose = 'capture')
    _, synthetic = features(pair, purpose = 'synthetic')
    torch.testing.assert_close(trace['gain'], torch.tensor([2., 3.]))
    assert torch.equal(synthetic['gain'], torch.tensor([9., 9.]))
    separate, _ = features(pair[:1], purpose = 'capture')
    assert torch.equal(capture[:1], separate)


@pytest.mark.parametrize('invalid', ['nan', 'range', 'dtype', 'shape', 'mask', 'patch'])
def test_invalid_inputs_fail_explicitly(invalid):
    features = Features(profile()['features'], device = 'cpu')
    pair = torch.full((1, 6, 256, 256), .02)
    mask = torch.ones(1, 1, 256, 256, dtype = torch.bool)
    if invalid == 'nan':
        pair[0, 0, 0, 0] = float('nan')
    elif invalid == 'range':
        pair[0, 0, 0, 0] = -1
    elif invalid == 'dtype':
        pair = pair.double()
    elif invalid == 'shape':
        pair = pair[:, :3]
    elif invalid == 'mask':
        mask[:] = False
    elif invalid == 'patch':
        mask[:, :, 128, 128] = False
    with pytest.raises(ValueError):
        features(pair, purpose = 'capture', mask = mask)


def test_spatial_factor_and_legacy_profile_are_explicit():
    config = profile()['features']
    spatial = Features(config, device = 'cpu')
    config = {**config, 'relation': 'legacy_no_spatial'}
    legacy = Features(config, device = 'cpu')
    assert spatial.spatial[0, 0, 127, 127] == 1
    assert spatial.spatial[0, 0, 0, 0] < 1
    assert torch.equal(legacy.spatial, torch.ones_like(legacy.spatial))


def test_normal_loss_and_invalid_support():
    raw = torch.zeros(1, 10, 2, 2)
    raw[:, 2] = 2
    pred = decode(raw, 'unit_xyz')
    assert torch.equal(pred[:, 2], torch.ones_like(pred[:, 2]))
    target = pred.clone()
    safe, mask = canonical_target(target, roughness_floor = .05)
    assert map_loss(safe, safe, mask)[0] == 0
    with pytest.raises(ValueError):
        map_loss(pred, target, torch.zeros_like(mask))
    with pytest.raises(ValueError):
        decode(torch.zeros_like(raw), 'unit_xyz')


def test_weight_normalization_rejects_collision_and_nonfinite():
    with pytest.raises(ValueError):
        state_dict({'module.x': torch.ones(1), 'x': torch.ones(1)})
    with pytest.raises(ValueError):
        state_dict({'x': torch.tensor([float('inf')])})


def make_trainer():
    policy = json.loads((ROOT / 'configs/training.json').read_text())['policy']
    runtime = json.loads((ROOT / 'configs/runtime.json').read_text())
    runtime.update(sample_count = 8, batch_size = 2, disk_reserve_bytes = 0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.1), torch.nn.Linear(3, 1))
    def objective(network, indices, transforms, stage):
        x = torch.tensor(list(zip(indices, transforms)), dtype = torch.float32) / 8
        return (network(x) - .3).square().mean(), {}
    def evaluate(network, stage, step):
        return {'heldout': {'mean': {'ndrs_sum': 1., 'render_l1': .1}}}
    return Trainer(net, {'policy': policy}, objective, evaluate, runtime)


def equal_tree(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


def test_resume_repeats_next_batch_dropout_adam_and_scheduler():
    original = make_trainer()
    original.step()
    original.evaluate()
    saved = original.snapshot()
    expected_row = original.step()
    original.evaluate()
    expected = original.snapshot()
    resumed = make_trainer()
    resumed.restore(saved)
    assert resumed.step() == expected_row
    resumed.evaluate()
    equal_tree(expected, resumed.snapshot())
    saved['relation_version'] = 'legacy_no_spatial'
    with pytest.raises(ValueError):
        resumed.restore(saved)


def test_evaluation_preserves_rng_and_training_modes():
    trainer = make_trainer()
    trainer.net.train()
    trainer.net[1].eval()
    before = capture_rng()
    with evaluation_mode(trainer.net):
        torch.rand(5)
        assert not trainer.net.training
    equal_tree(before, capture_rng())
    assert trainer.net.training and not trainer.net[1].training


def test_budget_pauses_without_forcing_phase(tmp_path):
    trainer = make_trainer()
    state = trainer.run(tmp_path, {'map_cap': 1, 'render_cap': 1})
    assert state['status'] == 'budget_paused' and state['stage'] == 'map'
    assert state['render_updates'] == 0


def test_atomic_failure_preserves_existing_file(tmp_path):
    path = tmp_path / 'saved'
    path.write_bytes(b'original')
    def fail(stream):
        stream.write(b'partial')
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        atomic_write(path, fail)
    assert path.read_bytes() == b'original'
    with output_lock(tmp_path):
        with pytest.raises(FileExistsError):
            with output_lock(tmp_path):
                pass


def test_burst_average_and_fallback_never_drop_failed_frames():
    manifest = {'frames': {side: [{'index': index, 'error': None, 'full_crop': True, 'eligible': True}
                                for index in range(5)] for side in ('near', 'far')}}
    review = {'status': 'reviewed', 'texture_registration_passed': True, 'note': 'checked'}
    assert select_frames(manifest, review)[0] == list(range(5))
    manifest['frames']['far'][4]['eligible'] = False
    assert select_frames(manifest, review)[0] == [0]
    manifest['frames']['near'][0]['full_crop'] = False
    with pytest.raises(ValueError):
        select_frames(manifest, review)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA unavailable')
def test_cuda_alias_accepts_input_on_resolved_device():
    config = json.loads((ROOT / 'configs/7m_spatial.json').read_text())
    features = Features(config['features'], device = 'cuda')
    pair = torch.full((1, 6, 256, 256), .02, device = 'cuda')
    result, _ = features(pair, purpose = 'capture')
    assert result.device == pair.device and result.shape == (1, 21, 256, 256)


@pytest.mark.parametrize('corruption', ['optimizer_lr', 'counter', 'scheduler'])
def test_resume_rejects_corrupt_persistent_state(corruption):
    trainer = make_trainer()
    trainer.step()
    payload = trainer.snapshot()
    if corruption == 'optimizer_lr':
        payload['optimizer']['param_groups'][0]['lr'] = float('nan')
    elif corruption == 'counter':
        payload['state']['total_step'] = -1
    else:
        payload['scheduler']['best'] = float('nan')
    with pytest.raises(ValueError):
        trainer.restore(payload)
