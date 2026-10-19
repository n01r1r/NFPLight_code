import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

from nfplight.checkpoint import read_config, load_payload, state_dict, check_profile, decode
from nfplight.features import Features
from nfplight.network import NFPLight
from nfplight.rendering import Renderer
from nfplight.storage import identity, write_json
from nfplight.training import Trainer

ROOT = Path(__file__).resolve().parent


def equal(left, right, label):
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            raise ValueError(f'{label}: tensor mismatch (max abs {(left - right).abs().max().item()})')
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f'{label}: different keys')
        for key in left:
            equal(left[key], right[key], f'{label}.{key}')
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f'{label}: different lengths')
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, f'{label}[{index}]')
    elif left != right:
        raise ValueError(f'{label}: {left} != {right}')


def verify(legacy, output, device = 'cpu'):
    legacy = Path(legacy).resolve()
    sys.path.insert(0, str(legacy))
    from network.nfplight_net import TwoBranchNet, NAFBlock
    from model.nfplight_model import NFPLightModel
    import model.author_real_capture_adapter as author
    from train_fabric import render_input_images_batched, render_batch
    from experiments.full_dataset_suite.engine import Engine
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(17)
    class Reference(NFPLightModel):
        def init_network(self):
            pass
    paths = sorted((legacy / 'artifacts/weight_handoff_20261019/weights').glob('*.pt'))
    if len(paths) != 3:
        raise ValueError('expected the three delivered 7M checkpoints')
    cases = list(zip(paths, ('7m_original', '7m_spatial', '7m_c')))
    cases += [(legacy / 'artifacts/C_convergence_followup_20261019/run/last.pt', '7m_c'),
              (legacy / 'checkpoints/net_g_real.pth', 'author_real'),
              (legacy / 'checkpoints/net_g_syn.pth', 'author_syn')]
    results = []
    pair = torch.rand(1, 6, 256, 256, device = device) * .05
    for path, name in cases:
        before = identity(path)
        payload = load_payload(path)
        if identity(path) != before:
            raise ValueError('checkpoint changed while reading; retry with a stable snapshot')
        config = read_config(ROOT / f'configs/{name}.json')
        check_profile(payload, config)
        old_type = author.AuthorTwoBranchRealNet if name == 'author_real' else TwoBranchNet
        architecture = config['network']
        old = old_type(width = architecture['dim'], middle_blk_num = architecture['middle_depth'],
            enc_blk_nums = architecture['encoder_depths'], dec_blk_nums = architecture['decoder_depths'])
        for module in old.modules():
            if isinstance(module, NAFBlock):
                module.dropout1 = torch.nn.Dropout(architecture['dropout'])
                module.dropout2 = torch.nn.Dropout(architecture['dropout'])
        new = NFPLight(**architecture)
        equal(list(dict(old.named_parameters())), list(dict(new.named_parameters())), 'parameter order')
        weights = state_dict(payload)
        old.load_state_dict(weights, strict = True)
        new.load_state_dict(weights, strict = True)
        old.to(device).eval()
        new.to(device).eval()
        features = Features(config['features'], device = device)
        actual_features, _ = features(pair, purpose = 'capture')
        if name == 'author_real':
            coefficient, angular, maximum = author.legacy_geometry(256, device)
            original_factor = author.planar_spatial_factor
            try:
                author.planar_spatial_factor = lambda *args: torch.ones_like(coefficient)
                expected_features = author.assemble_legacy33_features(pair, pair, coefficient, angular, maximum)[0]
            finally:
                author.planar_spatial_factor = original_factor
        else:
            reference = Reference(SimpleNamespace(device = device, feature_version = config['feature_version'],
                capture_precision = config['features']['geometry_dtype'] == 'float32'))
            if config['features']['relation'] == 'legacy_no_spatial':
                reference.spatial_attenuation_factor = torch.ones_like(reference.spatial_attenuation_factor)
            expected_features = reference.build_capture_features(pair)[0]
            if name.startswith('7m'):
                equal(reference.build_features(pair)[0], features(pair, purpose = 'synthetic')[0], 'synthetic features')
        equal(expected_features, actual_features, 'capture features')
        with torch.inference_mode():
            expected = old(expected_features)
            actual = new(actual_features)
        equal(expected, actual, 'full 256px forward')
        equal(decode(expected, config['decoder']), decode(actual, config['decoder']), 'decoded output')
        results.append({'checkpoint': before, 'profile': name, 'parameter_order_exact': True,
                        'strict_load': True, 'features_exact': True, 'prediction_exact': True})
        print('PASS', name, path.name, flush = True)
        del old, new, payload, weights, expected, actual
        gc.collect()

    profile = read_config(ROOT / 'configs/7m_spatial.json')
    runtime = json.loads((ROOT / 'configs/runtime.json').read_text())
    reference = Reference(SimpleNamespace(device = device, feature_version = profile['feature_version'], capture_precision = True))
    renderer = Renderer(runtime['renderer'], device = device)
    maps = torch.rand(1, 10, 256, 256, device = device) * 2 - 1
    equal(render_input_images_batched(reference, maps), renderer.observations(maps, 4., 12.), 'observation rendering')
    lights = torch.tensor([[0., 0., 4.], [1., 1., 3.]], device = device)
    views = torch.tensor([[0., 0., 2.75], [0., 0., 2.75]], device = device)
    equal(render_batch(reference, maps, lights, views), renderer(maps, lights, views), 'novel rendering')
    policy = json.loads((ROOT / 'configs/training.json').read_text())['policy']
    runtime.update(sample_count = 8, batch_size = 2, disk_reserve_bytes = 0)
    identity_record = {'policy': policy}
    def objective(network, indices, transforms, stage):
        x = torch.tensor(list(zip(indices, transforms)), dtype = torch.float32) / 8
        return (network(x) - .3).square().mean(), {}
    def evaluator(network, stage, step):
        return {'heldout': {'mean': {'ndrs_sum': 1., 'render_l1': .1}}}
    network = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.1), torch.nn.Linear(3, 1))
    new = Trainer(network, identity_record, objective, evaluator, runtime)
    new.step()
    new.evaluate()
    initial = new.snapshot()
    new.step()
    new.state['status'] = 'interrupted'
    expected = new.snapshot()
    with tempfile.TemporaryDirectory() as folder:
        old = Engine(folder, copy.deepcopy(network), policy, identity_record, objective, evaluator, sample_count = 8, batch_size = 2)
        torch.save(initial, Path(folder) / 'last.pt')
        old.run({'map_cap': 10, 'render_cap': 10}, resume = True, interrupt_after = 2)
        equal(expected, old.snapshot(), 'legacy engine next update including all states')
        new.evaluate()
        old.evaluate()
        equal(new.snapshot(), old.snapshot(), 'legacy evaluation/scheduler state')
    report = {'passed': True, 'device': device, 'engine_fixture_device': 'cpu', 'torch': str(torch.__version__), 'cases': results,
        'renderers_exact': True, 'legacy_engine_next_update_exact': True,
        'production_gpu_continuation_tested': False, 'sha256_computed': False}
    write_json(output, report)
    print('PASS renderers and legacy engine exact next update; report:', output, flush = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Read-only legacy comparison; writes only a new verification report')
    parser.add_argument('--legacy-root', required = True)
    parser.add_argument('--out', required = True)
    parser.add_argument('--device', choices = ('cpu', 'cuda'), default = 'cpu')
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError('verification output already exists')
    verify(args.legacy_root, args.out, args.device)
