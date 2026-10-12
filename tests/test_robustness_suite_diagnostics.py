from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.robustness_suite import diagnostics as d


def inputs():
    values = torch.full((1, 6, 10, 10), 0.5, dtype=torch.float32)
    valid = torch.ones((1, 1, 10, 10), dtype=torch.bool)
    return values, valid


class FeatureModel:
    feature_version = 'raw_calibrated_gain9_v3'

    def build_features(self, observations, valid_mask):
        self.observations = observations.clone()
        self.valid = valid_mask.clone()
        # The real builder receives raw observations before applying fixed gain9.
        return torch.cat((observations, observations[:, 3:] * 9), dim=1), None


def test_noise_zero_identity_preserves_support_and_rng(monkeypatch):
    observation, valid = inputs()
    valid[..., 0, 0] = False
    target = torch.zeros((1, 10, 10, 10), dtype=torch.float32)
    model = FeatureModel()
    monkeypatch.setattr(d, 'render_input_images_batched', lambda model, target: observation)
    generator = d.make_noise_generator(19)
    before = generator.get_state().clone()
    features = d.noise_features(model, target, valid, generator, {'sigma': 0})
    assert torch.equal(model.observations, observation * valid)
    assert torch.equal(model.valid, valid)
    assert torch.equal(features[:, 6:], observation[:, 3:] * valid * 9)
    assert torch.equal(generator.get_state(), before)


def test_seed_restore_and_global_rng_neutrality(monkeypatch):
    observation, valid = inputs()
    target = torch.zeros((1, 10, 10, 10), dtype=torch.float32)
    monkeypatch.setattr(d, 'render_input_images_batched', lambda model, target: observation)
    generator = d.make_noise_generator(19)
    saved = generator.get_state().clone()
    global_state = torch.get_rng_state().clone()
    first = d.noise_features(FeatureModel(), target, valid, generator, {'sigma': .005})
    second = d.noise_features(FeatureModel(), target, valid, generator, {'sigma': .005})
    assert not torch.equal(first, second)
    generator.set_state(saved)
    assert torch.equal(first, d.noise_features(FeatureModel(), target, valid, generator, {'sigma': .005}))
    assert torch.equal(first, d.noise_features(FeatureModel(), target, valid, d.make_noise_generator(19), {'sigma': .005}))
    assert not torch.equal(first, d.noise_features(FeatureModel(), target, valid, d.make_noise_generator(20), {'sigma': .005}))
    assert torch.equal(global_state, torch.get_rng_state())


@pytest.mark.parametrize('sigma', [.001, .005, .01])
def test_gaussian_signed_linear_severity(sigma):
    observation, valid = inputs()
    result = d.perturb_observations(observation, valid, {'axis': 'gaussian', 'sigma': sigma}, d.make_noise_generator(19))
    difference = result - observation
    assert difference.min() < 0 < difference.max()
    base = d.perturb_observations(observation, valid, {'axis': 'gaussian', 'sigma': .001}, d.make_noise_generator(19)) - observation
    assert torch.allclose(difference, base * (sigma / .001), atol=3e-7)
    assert result.min() >= 0 and result.max() <= 1


def test_clip_and_pixel_support_are_explicit():
    observation, valid = inputs()
    observation[..., :5, :] = 0
    observation[..., 5:, :] = 1
    valid[..., 0, 0] = False
    mask = valid.clone()
    result = d.perturb_observations(observation, valid, {'axis': 'gaussian', 'sigma': .01}, d.make_noise_generator(19))
    assert result.min() == 0 and result.max() == 1
    assert (result[..., 0, 0] == 0).all()
    assert torch.equal(valid, mask)


def test_pixel_dropout_exact_fraction_same_near_far_and_deterministic():
    observation, valid = inputs()
    condition = {'axis': 'pixel_dropout', 'fraction': .01}
    first = d.perturb_observations(observation, valid, condition, d.make_noise_generator(19))
    second = d.perturb_observations(observation, valid, condition, d.make_noise_generator(19))
    assert torch.equal(first, second)
    assert (first[0, 0] == 0).sum() == 1
    for channel in first[0]:
        assert torch.equal(channel, first[0, 0])
    assert valid.all()


def test_saturation_multiplies_only_raw_observations():
    observation, valid = inputs()
    observation[:, :3] = .95
    result = d.perturb_observations(observation, valid, {'axis': 'saturation', 'multiplier': 1.2}, d.make_noise_generator(19))
    assert (result[:, :3] == 1).all()
    assert torch.allclose(result[:, 3:], torch.full_like(result[:, 3:], .6))


@pytest.mark.parametrize('condition', [
    {'axis': 'gaussian', 'sigma': -.1}, {'axis': 'gaussian', 'sigma': float('nan')},
    {'axis': 'gaussian', 'sigma': .005, 'fraction': .01}, {'axis': 'unknown'},
    {'axis': 'pixel_dropout', 'fraction': 1.1}, {'axis': 'saturation', 'multiplier': .9},
])
def test_invalid_stress_rejected(condition):
    observation, valid = inputs()
    with pytest.raises(ValueError):
        d.perturb_observations(observation, valid, condition, d.make_noise_generator(19))


@pytest.mark.parametrize('case', ['nan', 'range', 'dtype', 'mask', 'empty'])
def test_invalid_input_rejected(case):
    observation, valid = inputs()
    if case == 'nan':
        observation[..., 0, 0] = float('nan')
    elif case == 'range':
        observation[..., 0, 0] = -1
    elif case == 'dtype':
        observation = observation.double()
    elif case == 'mask':
        valid = valid.float()
    else:
        valid.fill_(False)
    with pytest.raises((ValueError, FloatingPointError)):
        d.perturb_observations(observation, valid, {'axis': 'clean'}, d.make_noise_generator(19))


def test_representation_contract_rejected(monkeypatch):
    target = torch.zeros((1, 10, 10, 10), dtype=torch.float32)
    _, valid = inputs()
    with pytest.raises(ValueError, match='fixed FP32'):
        d.noise_features(FeatureModel(), target, valid, d.make_noise_generator(19), {'rendering': {'gain': 10}})
    model = FeatureModel()
    model.feature_version = 'sample_v1'
    with pytest.raises(ValueError, match='raw_calibrated'):
        d.noise_features(model, target, valid, d.make_noise_generator(19), {})


def losses(parameter, render=None):
    normal = parameter.sum()
    diffuse = parameter[0] - parameter[1]
    roughness = 2 * parameter[0]
    return {'normal': normal, 'diffuse': diffuse, 'roughness': roughness,
            'map': (normal + diffuse + roughness) / 3,
            'render': 3 * parameter[1] if render is None else render}


def test_analytic_gradient_norm_cosine_and_grad_unmodified():
    parameter = nn.Parameter(torch.tensor([2., 3.]))
    parameter.grad = torch.tensor([9., 8.])
    previous = parameter.grad.clone()
    components = losses(parameter)
    metrics = d.gradient_metrics(components, [('p', parameter)])
    assert metrics['finite']
    assert metrics['norms'] == pytest.approx({'normal': 2 ** .5, 'diffuse': 2 ** .5, 'roughness': 2, 'map': 4 / 3, 'render': 3})
    assert metrics['map_render_cosine'] == 0
    assert torch.equal(parameter.grad, previous)
    (components['map'] + components['render']).backward()
    assert torch.allclose(parameter.grad, previous + torch.tensor([4 / 3, 3.]))
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_analytic_parallel_cosine(sign):
    parameter = nn.Parameter(torch.tensor([2., 3.]))
    components = losses(parameter)
    components['render'] = sign * components['map']
    metrics = d.gradient_metrics(components, [('p', parameter)])
    assert metrics['map_render_cosine'] == pytest.approx(sign)
    assert parameter.grad is None


def test_inactive_and_zero_render_are_undefined():
    parameter = nn.Parameter(torch.tensor([2., 3.]))
    metrics = d.gradient_metrics(losses(parameter, torch.tensor(0.)), [('p', parameter)])
    assert metrics['norms']['render'] is None
    assert metrics['map_render_cosine'] is None
    assert 'render' in metrics['undefined']
    metrics = d.gradient_metrics(losses(parameter, parameter.sum() * 0), [('p', parameter)])
    assert metrics['norms']['render'] == 0
    assert metrics['map_render_cosine'] is None


def test_unused_parameters_and_nonfinite_gradients():
    parameter = nn.Parameter(torch.tensor([2., 3.]))
    unused = nn.Parameter(torch.tensor([4.]))
    metrics = d.gradient_metrics(losses(parameter), [('p', parameter), ('unused', unused)])
    assert metrics['trainable_parameters'] == 3
    assert unused.grad is None
    zero = nn.Parameter(torch.tensor([0., 1.]))
    with pytest.raises(FloatingPointError, match='gradient'):
        d.gradient_metrics(losses(zero, zero[0].sqrt()), [('p', zero)])
    components = losses(parameter)
    components['normal'] = parameter.sum() * float('nan')
    with pytest.raises(FloatingPointError, match='loss'):
        d.gradient_metrics(components, [('p', parameter)])


def clean_rows():
    return {name: [{**{metric: .25 for metric in d.METRICS}, 'material_id': f'id{i}', 'category': 'Fabric', 'crop': 4}
                   for i in range(8)] for name in ('heldout', 'unseen_crop')}


def test_clean_reproduction_every_metric_and_identity():
    actual = clean_rows()
    expected = {key: {'materials': copy.deepcopy(rows)} for key, rows in actual.items()}
    assert d.assert_clean_reproduction(actual, expected)['passed']
    actual['unseen_crop'][7]['near_l1'] = 100
    with pytest.raises(AssertionError, match='near_l1'):
        d.assert_clean_reproduction(actual, expected)
    actual = clean_rows()
    actual['heldout'][0]['crop'] = 0
    with pytest.raises(AssertionError, match='identities'):
        d.assert_clean_reproduction(actual, expected)


def test_disagreement_is_proxy_and_constant_correlation_undefined():
    left = torch.zeros((1, 10, 2, 2))
    right = left + .25
    valid = torch.ones((1, 1, 2, 2), dtype=torch.bool)
    assert d.prediction_disagreement(left, right, valid) == {'normal': .25, 'diffuse': .25, 'roughness': .25}
    assert d._correlation([(1, 3), (1, 4)]) is None
    assert d._correlation([(1, 3), (2, 4)]) == pytest.approx(1)


def test_read_only_evaluation_cpu_orchestration_selection_and_all_stress(tmp_path, monkeypatch):
    import experiments.regularization.run as old
    from experiments.regularization.config import file_identity

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.))

        def forward(self, features):
            shape = (len(features), 1, *features.shape[-2:])
            zero = features[:, :1] * 0 + self.bias
            return torch.cat((zero, zero, torch.ones(shape), zero.expand(-1, 7, -1, -1)), 1)

    class Model(FeatureModel):
        def __init__(self, settings, dropout, device):
            self.net_g = Net()
            self.device = torch.device('cpu')

    class Bank:
        def __init__(self, manifest, dataset):
            self.cache = {}

        def crop(self, slot):
            return torch.zeros((10, 10, 10), dtype=torch.float32)

    monkeypatch.setattr(old, 'RegularizationModel', Model)
    monkeypatch.setattr(d, 'Bank', Bank)
    monkeypatch.setattr(d, '_memory_guard', lambda: None)
    monkeypatch.setattr(d, 'prepare_features', lambda model, raw, f0: (raw, inputs()[1], inputs()[0]))
    monkeypatch.setattr(d, 'render_input_images_batched', lambda model, raw: inputs()[0])
    monkeypatch.setattr(d, '_metrics', lambda *args: {**{key: .25 for key in d.METRICS}, 'novel_light_l1': [.25] * 12})
    manifest = tmp_path / 'manifest.jsonl'
    shard = tmp_path / 'fake.npz'
    manifest.write_text('{}\n')
    shard.write_text('read-only CPU test input')
    slots = [{**row, 'shard_path': str(shard)} for row in clean_rows()['heldout']]
    execution = {'scientific_settings': {'dataset': {}, 'backbone': {},
        'rendering': {'near_distance': 4., 'far_distance': 12., 'gain': 9., 'feature_version': 'raw_calibrated_gain9_v3',
                      'precision': 'float32', 'f0': .04, 'novel_lights': {'theta_degrees': [12, 27, 40],
                          'phi_degrees': [22.5, 112.5, 202.5, 292.5], 'radius': 4.}},
        'runtime': {'device': 'cpu', 'torch_num_threads': 1, 'cudnn_benchmark': False, 'cudnn_deterministic': True}},
        'resolved_inputs': {'manifest': str(manifest), 'heldout': slots, 'unseen_crop': slots},
        'file_identities': [file_identity(manifest), file_identity(shard)]}
    execution_path = tmp_path / 'execution.json'
    execution_path.write_text(json.dumps(execution))
    reference = tmp_path / 'evaluation.json'
    reference.write_text(json.dumps([
        {'stage': 'map', 'step': 0, 'total_step': 0, 'populations': {}},
        {'stage': 'render', 'step': 0, 'total_step': 500,
         'populations': {key: {'materials': rows} for key, rows in clean_rows().items()}},
    ]))
    specs = []
    for label in ('map500', 'final'):
        checkpoint = tmp_path / (label + '.pt')
        torch.save({'net_g': Net().state_dict(), 'identity': execution,
                    'stage': 'render', 'step': 0, 'total_step': 500}, checkpoint)
        specs.append({'label': label, 'checkpoint': checkpoint, 'execution': execution_path, 'reference': reference})
    before = [file_identity(path) for path in (manifest, shard, reference, execution_path, *(spec['checkpoint'] for spec in specs))]
    result = d.evaluate(tmp_path / 'out', checkpoint_specs=specs)
    assert set(result) == {'map500', 'final'}
    assert all(record['clean_reproduction']['passed'] for record in result.values())
    assert all(len(record['conditions']) == 6 for record in result.values())
    assert all(len(record['conditions']['clean'][name]) == 8 for record in result.values() for name in ('heldout', 'unseen_crop'))
    complete = json.loads((tmp_path / 'out/complete.json').read_text())
    assert complete['source_identities_unchanged']
    disagreement = json.loads((tmp_path / 'out/disagreement.json').read_text())
    assert len(disagreement['rows']) == 96
    assert [file_identity(row['path']) for row in before] == before
    with pytest.raises(FileExistsError):
        d.evaluate(tmp_path / 'out', checkpoint_specs=specs)


def test_suite_specs_use_own_execution_and_selected_reference(tmp_path):
    directory = tmp_path / 'baseline7m_p01'
    directory.mkdir()
    (directory / 'complete.json').write_text('{}')
    spec = d._suite_specs(tmp_path)[0]
    assert spec['execution'] == directory / 'execution_config.json'
    assert spec['checkpoint'] == directory / 'selected.pt'
    assert 'reference_index' not in spec
