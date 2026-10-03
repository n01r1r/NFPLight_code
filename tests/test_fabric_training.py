"""Deterministic CPU regression tests for Fabric contracts and resume behavior."""
import copy
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from data.fabric import (
    FabricDataset,
    canonical_target,
    decode_tile,
    file_hash,
    prepare,
    write_json,
)
from model.nfplight_model import NFPLightModel, checkpoint_feature_version
from train_fabric import (
    FEATURE_VERSION,
    LEGACY_L1_TRAIN_SOURCE_SHA256,
    FabricObservationCache,
    _atomic_link_or_copy,
    _assert_transition_contract,
    _checkpoint_due,
    _cleanup_checkpoint_candidates,
    _commit_last,
    _render_restart_state,
    _rewrite_selection_candidate,
    _sample_batch,
    _validate_performance_migration,
    _validate_render_restart_source,
    _validate_render_schedule_migration,
    _write_candidate,
    augmentation_rng,
    build_contract,
    cosine_lr,
    evaluate,
    load_checkpoint,
    map_loss,
    optimizer_step,
    parse_args,
    render_batch,
    render_candidate,
    render_input_images_batched,
    render_pair,
    save_checkpoint,
)
from train_fabric import render_loss as fabric_render_loss
from utils import svBRDF


def cpu_model():
    model = NFPLightModel.__new__(NFPLightModel)
    model.device = torch.device('cpu')
    model.feature_version = 'sample_v1'
    model.renderer = svBRDF()
    model.surface = model.renderer.surface(256, 1)
    for prefix, z in [('near', 2.414), ('far', 10)]:
        position = torch.tensor([[0., 0., z]])
        direction, _, distance, _ = model.renderer.torch_generate(position, position, pos=model.surface)
        setattr(model, prefix + '_light_dir', direction)
        setattr(model, prefix + '_light_dis', distance)
    model.coefficient = torch.linspace(0, 1, 256 * 256).reshape(1, 1, 256, 256)
    model.time_co_map = torch.ones_like(model.coefficient) * 0.9
    model.indenty = torch.ones_like(model.coefficient)
    model.net_g = nn.Sequential(nn.Conv2d(21, 10, 1), nn.Tanh())
    return model


def write_material(root, split, name, number, *, source='test_source', opacity=255, declared=True):
    path = root / split / 'Fabric' / name
    path.mkdir(parents=True)
    arrays = {
        'normal': np.full((260, 260, 3), [128 + number, 125, 255], dtype=np.uint8),
        'diffuse': np.full((260, 260, 3), 128, dtype=np.uint8),
        'roughness': np.full((260, 260), 160 + number, dtype=np.uint8),
        'specular': np.full((260, 260, 3), 59, dtype=np.uint8),
        'opacity': np.full((260, 260), opacity, dtype=np.uint8),
    }
    for key, pixels in arrays.items():
        Image.fromarray(pixels).save(path / f'{key}.png')
    metadata = {'category': 'Fabric', 'source': source, 'link': 'https://source.example/',
                'maps': list(arrays) if declared else [key for key in arrays if key != 'opacity']}
    write_json(path / 'metadata.json', {'metadata': metadata})
    return path


class FabricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / 'source'
        cls.output = Path(cls.temporary.name) / 'cache'
        for index in range(4):
            write_material(cls.root, 'train', f'fabric_{index}', index)
        write_material(cls.root, 'train', 'color_variant', 0)
        write_material(cls.root, 'train', 'test_variant', 9)
        write_material(cls.root, 'train', 'transparent', 6, opacity=0)
        write_material(cls.root, 'train', 'excluded_source', 7, source='deschaintre_2020')
        write_material(cls.root, 'test', 'official_test', 9, declared=False)
        cls.manifest = prepare(cls.root, cls.output, crop_count=2, seed=0, val_ratio=0.25)
        cls.path = cls.output / 'manifest.json'

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_manifest_exclusions_groups_native_crops_and_rebuild(self):
        reasons = [row['reason'] for row in self.manifest['rejected']]
        self.assertIn('excluded_source_deschaintre_2020', reasons)
        self.assertIn('shares_group_with_official_test', reasons)
        self.assertTrue(any(reason.startswith('nonopaque') for reason in reasons))
        records = {row['name']: row for row in self.manifest['records']}
        self.assertEqual(records['fabric_0']['split'], records['color_variant']['split'])
        row = records['fabric_1']
        pixels = np.load(self.output / row['cache'])
        source = np.asarray(Image.open(self.root / row['id'] / 'diffuse.png'))
        y, x = row['coordinates_yx'][0]
        np.testing.assert_array_equal(pixels[0, :, :, 3:6], source[y:y+256, x:x+256])
        previous = file_hash(self.path)
        prepare(self.root, self.output, crop_count=2, seed=0, val_ratio=0.25)
        self.assertEqual(previous, file_hash(self.path))

    def test_gamma_normal_and_all_d4_transforms(self):
        dataset = FabricDataset(self.path, split='train')
        tile = dataset.crops[0][0]
        base = decode_tile(tile)
        self.assertAlmostEqual(float(base[3, 0, 0]), (128 / 255)**2.2 * 2 - 1, places=6)
        self.assertAlmostEqual(float(base[7, 0, 0]), (59 / 255)**2.2 * 2 - 1, places=6)
        original = base[:3, 0, 0].numpy()
        for rotation in range(4):
            angle = rotation * np.pi / 2
            matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            for flip in [False, True]:
                sv = decode_tile(tile, rotation=rotation, hflip=flip)
                expected = matrix @ original[:2]
                if flip:
                    expected[0] *= -1
                np.testing.assert_allclose(sv[:2, 0, 0], expected, atol=1e-7)
                torch.testing.assert_close(sv[:3].norm(dim=0), torch.ones(256, 256))

    def test_dataset_rejects_leakage_and_tampered_cache(self):
        bad = copy.deepcopy(self.manifest)
        bad['records'][0]['split'] = 'test' if bad['records'][0]['split'] != 'test' else 'train'
        duplicate = copy.deepcopy(bad['records'][0])
        duplicate['id'] += '_duplicate'
        duplicate['split'] = 'val'
        bad['records'].append(duplicate)
        bad_path = self.output / 'bad.json'
        write_json(bad_path, bad)
        with self.assertRaisesRegex(ValueError, 'leaks'):
            FabricDataset(bad_path, split='train')
        bad = copy.deepcopy(self.manifest)
        row = next(row for row in bad['records'] if row['split'] == 'train')
        row['cache_sha256'] = 'bad hash'
        write_json(bad_path, bad)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            FabricDataset(bad_path, split='train')

    def test_input_batch_invariance_dark_saturated_and_legacy(self):
        model = cpu_model()
        generator = torch.Generator().manual_seed(9)
        pair = torch.rand(1, 6, 256, 256, generator=generator) * 0.7
        other = torch.rand(1, 6, 256, 256, generator=generator)
        alone = model.build_features(pair)[0]
        batched = model.build_features(torch.cat((pair, other)))[0][:1]
        torch.testing.assert_close(alone, batched, rtol=0, atol=1e-6)
        for inputs in [torch.zeros_like(pair), torch.ones_like(pair)]:
            self.assertTrue(torch.isfinite(model.build_features(inputs)[0]).all())
        with self.assertRaisesRegex(ValueError, 'finite'):
            model.build_features(torch.full_like(pair, float('nan')))
        model.feature_version = 'legacy_batch_v0'
        feature = torch.tensor([[[[1., 2.], [3., 4.]]]])
        self.assertGreater(float((model.ClipToOne(feature) - model.ClipToOne(torch.cat((feature, feature * 10)))[:1]).abs().max()), 0.9)
        self.assertEqual(checkpoint_feature_version({}), 'legacy_batch_v0')
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            checkpoint_feature_version({'feature_version': 'unknown'})

    def test_batched_renderer_matches_fixed_probe_paths(self):
        model = cpu_model()
        target = FabricDataset(self.path, split='train').get_crop(0, 0)[None]
        expected_input = model.render_input_images(target)
        actual_input = render_input_images_batched(model, target)
        torch.testing.assert_close(actual_input, expected_input, rtol=1e-5, atol=1e-6)
        lights = torch.tensor([[0., 0., 3.], [0.5, -0.5, 4.]])
        actual = render_batch(model, target, lights, lights)[0]
        for index, light in enumerate(lights):
            expected = render_pair(model, target, light, light)[0]
            torch.testing.assert_close(actual[index], expected, rtol=1e-5, atol=1e-6)

    def test_vectorized_render_loss_matches_scalar_reduction(self):
        model = cpu_model()
        target = FabricDataset(self.path, split='train').get_crop(0, 0)[None]
        prediction = (target * 0.93).clamp(-1, 1)
        valid = torch.ones(1, 1, 256, 256, dtype=torch.bool)
        valid[..., :8, :8] = False
        lights = torch.tensor([[0., 0., 3.], [0.5, -0.5, 4.]])
        actual = fabric_render_loss(model, prediction, target, 2, lights, valid)
        support = valid.expand(-1, 3, -1, -1).sum()
        view = torch.tensor([0., 0., 2.75])
        expected = sum(
            (render_pair(model, prediction, light, view)
             - render_pair(model, target, light, view)).abs().masked_select(
                 valid.expand(-1, 3, -1, -1)
             ).sum() / support
            for light in lights
        ) / len(lights)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_observation_cache_is_lossless_and_material_granular(self):
        model = cpu_model()
        dataset = FabricDataset(self.path, split='train')
        with tempfile.TemporaryDirectory() as temporary:
            cache = FabricObservationCache(Path(temporary), dataset, 'train')
            cache.ensure_material(model, 0)
            slots = [(0, 0, 0, False), (0, 1, 0, False)]
            actual = cache.load_slots(slots)
            expected_targets = torch.stack([dataset.get_crop(0, crop) for crop in (0, 1)])
            expected_targets, _ = canonical_target(expected_targets)
            expected = render_input_images_batched(model, expected_targets).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(len(list((Path(temporary) / 'train').glob('*.npy'))), 1)
            for rotation in range(4):
                for hflip in (False, True):
                    slot = (0, 0, rotation, hflip)
                    cached = cache.load_slots([slot])
                    transformed = dataset.get_crop(
                        0, 0, rotation=rotation, hflip=hflip
                    )[None]
                    safe, _ = canonical_target(transformed)
                    directly_rendered = render_input_images_batched(model, safe).cpu()
                    torch.testing.assert_close(
                        cached, directly_rendered, rtol=1e-5, atol=1e-6
                    )

    def test_fused_accumulation_preserves_group_reduction(self):
        torch.manual_seed(31)
        separate = cpu_model()
        fused = cpu_model()
        fused.net_g.load_state_dict(copy.deepcopy(separate.net_g.state_dict()))
        dataset = FabricDataset(self.path, split='train')
        separate_optimizer = torch.optim.Adam(separate.net_g.parameters(), lr=1e-4)
        fused_optimizer = torch.optim.Adam(fused.net_g.parameters(), lr=1e-4)
        separate_rng = random.Random(41)
        fused_rng = random.Random(41)
        options = dict(batch_size=1, accumulate=2, weight=0.0, light_count=2)
        first = optimizer_step(separate, dataset, separate_optimizer, separate_rng, **options)
        second = optimizer_step(fused, dataset, fused_optimizer, fused_rng,
                                **options, fuse_accumulation=True)
        for key in ('map_loss', 'render_loss', 'normal_loss', 'loss'):
            self.assertAlmostEqual(first[key], second[key], places=5)
        for actual, expected in zip(fused.net_g.parameters(), separate.net_g.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)

    def test_performance_migration_is_explicit_and_narrow(self):
        current = build_contract(parse_args(['--out', 'unused']), self.path)
        source = copy.deepcopy(current)
        source['source_sha256']['train_fabric.py'] = (
            '55f77955f105f8435dfbed19790d58e862a5c8dd4afdb5f5ad5a49b6384b1867')
        source['settings'].pop('map_only')
        _validate_performance_migration(source, current)
        rejected = copy.deepcopy(source)
        rejected['settings']['batch'] = current['settings']['batch'] + 1
        with self.assertRaisesRegex(ValueError, 'scientific or dataset'):
            _validate_performance_migration(rejected, current)

    def test_checkpoint_alias_uses_same_volume_link_when_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'generation.pt'
            destination = root / 'last.pt'
            source.write_bytes(b'checkpoint-bytes')
            _atomic_link_or_copy(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            if os.stat(source).st_dev == os.stat(destination).st_dev:
                self.assertGreaterEqual(os.stat(source).st_nlink, 2)

    def test_fixed_evaluation_preserves_training_rng_and_mode(self):
        torch.manual_seed(11)
        model = cpu_model()
        dataset = FabricDataset(self.path, split='val')
        before = torch.get_rng_state().clone()
        first = evaluate(model, dataset, crop_count=2, light_count=2)
        second = evaluate(model, dataset, crop_count=2, light_count=2)
        single = evaluate(model, dataset, crop_count=2, light_count=2, eval_batch_size=1)
        self.assertEqual(first, second)
        self.assertTrue(model.net_g.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(row['crops'] == 2 for row in first['materials']))
        for key, value in first['mean'].items():
            self.assertAlmostEqual(value, single['mean'][key], places=5)

    def test_optimizer_checkpoint_exact_next_step(self):
        torch.manual_seed(4)
        model = cpu_model()
        dataset = FabricDataset(self.path, split='train')
        optimizer = torch.optim.Adam(model.net_g.parameters(), lr=1e-4)
        rng = random.Random(17)
        aug_rng = augmentation_rng(17, 'd4')
        options = dict(batch_size=1, accumulate=2, weight=0.5, light_count=2,
                       augmentation='d4', augmentation_rng=aug_rng)
        optimizer_step(model, dataset, optimizer, rng, **options)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'resume.pt'
            contract = {'manifest_sha256': file_hash(self.path),
                        'settings': {'augmentation': 'd4'}}
            save_checkpoint(path, model, contract,
                            {'stage': 'render', 'step': 1000, 'total_step': 1000},
                            optimizer=optimizer, rng=rng, augmentation_rng=aug_rng)
            saved_aug_rng_state = copy.deepcopy(aug_rng.getstate())
            expected = optimizer_step(model, dataset, optimizer, rng, **options)
            restored = cpu_model()
            restored_optimizer = torch.optim.Adam(restored.net_g.parameters(), lr=1e-4)
            restored_rng = random.Random(99)
            restored_aug_rng = augmentation_rng(99, 'd4')
            load_checkpoint(path, restored, contract, optimizer=restored_optimizer,
                            rng=restored_rng, augmentation_rng=restored_aug_rng)
            self.assertEqual(restored_aug_rng.getstate(), saved_aug_rng_state)
            restored_options = {**options, 'augmentation_rng': restored_aug_rng}
            actual = optimizer_step(
                restored, dataset, restored_optimizer, restored_rng, **restored_options
            )
            self.assertEqual(expected, actual)
            for a, b in zip(model.net_g.parameters(), restored.net_g.parameters()):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                load_checkpoint(path, restored, {'manifest_sha256': 'changed'})

    def test_amp_flag_is_explicit_and_scaler_state_is_checkpointed(self):
        args = parse_args(['--out', 'unused', '--amp'])
        self.assertTrue(args.amp)
        contract = build_contract(args, self.path)
        self.assertTrue(contract['settings']['amp'])
        model = cpu_model()
        model.feature_version = FEATURE_VERSION
        optimizer = torch.optim.Adam(model.net_g.parameters(), lr=1e-4)
        rng = random.Random(19)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        state = {'stage': 'map', 'step': 1, 'total_step': 1, 'refs': {}}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'amp.pt'
            save_checkpoint(checkpoint, model, contract, state,
                            optimizer=optimizer, rng=rng, scaler=scaler)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertIn('amp_scaler', payload)
            restored = cpu_model()
            restored_optimizer = torch.optim.Adam(restored.net_g.parameters(), lr=1e-4)
            restored_rng = random.Random(20)
            restored_scaler = torch.amp.GradScaler('cuda', enabled=False)
            load_checkpoint(checkpoint, restored, contract,
                            optimizer=restored_optimizer, rng=restored_rng,
                            scaler=restored_scaler)
            self.assertEqual(restored_scaler.state_dict(), scaler.state_dict())

    def test_checkpoint_cadence_default_and_step_1000_resume_boundary(self):
        self.assertEqual(parse_args(['--out', 'unused']).checkpoint_every, 1000)
        self.assertEqual(parse_args(['--out', 'unused']).render_steps, 100000)
        self.assertEqual(parse_args(['--out', 'unused', '--checkpoint-every', '1']).checkpoint_every, 1)
        contract = build_contract(parse_args(['--out', 'unused']), self.path)
        self.assertEqual(contract['settings']['checkpoint_every'], 1000)
        self.assertFalse(_checkpoint_due(999, 100000, 1000))
        self.assertTrue(_checkpoint_due(1000, 100000, 1000))
        self.assertFalse(_checkpoint_due(1001, 100000, 1000))
        self.assertTrue(_checkpoint_due(1001, 1001, 1000))
        self.assertTrue(_checkpoint_due(1, 100000, 1))
        with self.assertRaisesRegex(ValueError, 'positive'):
            _checkpoint_due(1, 100000, 0)

    def test_render_restart_source_is_strict_and_resets_stage_state(self):
        args = parse_args(['--out', 'unused', '--render-steps', '100000'])
        contract = build_contract(args, self.path)
        source_contract = copy.deepcopy(contract)
        source_contract['settings']['render_steps'] = 25000
        source_contract['settings'].pop('normal_head')
        source_contract['normal_objective'].pop('head')
        source_contract['normal_objective'].pop('head_formula')
        source_contract['source_sha256']['train_fabric.py'] = (
            'bf93f8a85cad73b86e7b30bdcb6f0c9f021287e3553c0c2cee8c0b9bd77797fc')
        source_contract['source_sha256'].pop('model/normal_head.py')
        model = cpu_model()
        model.feature_version = FEATURE_VERSION
        baseline = {'map_mean': 0.1, 'render_l1': 0.01}
        state = {'stage': 'render', 'step': 0, 'total_step': 100000,
                 'baseline': baseline, 'refs': {}}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'stage_a.pth'
            save_checkpoint(source, model, source_contract, state, metrics=baseline)
            write_json(Path(temporary) / 'initial.json', {'status': 'before_training',
                        'scope': 'validation', 'evaluation': {'mean': {'map_mean': 0.2}}})
            payload, initial = _validate_render_restart_source(source, contract)
            restarted = _render_restart_state(payload)
            self.assertEqual((restarted['stage'], restarted['step'], restarted['total_step']),
                             ('render', 0, 100000))
            self.assertEqual(restarted['best_render'], baseline['render_l1'])
            self.assertEqual(restarted['history'], [])
            fresh_optimizer = torch.optim.Adam(model.net_g.parameters(), lr=1e-4)
            self.assertEqual(fresh_optimizer.param_groups[0]['lr'], 1e-4)
            bad = copy.deepcopy(payload)
            bad['state']['stage'] = 'map'
            bad_path = Path(temporary) / 'stage_a.step-00000000.pth'
            torch.save(bad, bad_path)
            with self.assertRaisesRegex(ValueError, 'step-0 render boundary'):
                _validate_render_restart_source(bad_path, contract)

    def test_render_schedule_migration_requires_exact_historical_boundary(self):
        args = parse_args(['--out', 'unused', '--normal-loss', 'phi_theta', '--render-steps', '100000'])
        contract = build_contract(args, self.path)
        source_contract = copy.deepcopy(contract)
        source_contract['settings']['render_steps'] = 25000
        source_contract['source_sha256']['train_fabric.py'] = (
            'e150e3518c433e8d68e65b0670a3766c6b1b9fd12a11c9adea432dacaf508661')
        model = cpu_model()
        model.feature_version = FEATURE_VERSION
        optimizer = torch.optim.Adam(model.net_g.parameters(), lr=5e-4)
        rng = random.Random(7)
        state = {'stage': 'map', 'step': 15, 'total_step': 15, 'baseline': None,
                 'best_render': None, 'refs': {}, 'history': []}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'last.pt'
            save_checkpoint(path, model, source_contract, state, optimizer=optimizer, rng=rng)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            _validate_render_schedule_migration(payload, contract)
            bad = copy.deepcopy(payload)
            bad['state']['stage'] = 'render'
            with self.assertRaisesRegex(ValueError, 'map-stage'):
                _validate_render_schedule_migration(bad, contract)

    def test_schedule_migration_rewrites_selection_outside_cleanup_glob(self):
        args = parse_args(['--out', 'unused', '--normal-loss', 'phi_theta', '--render-steps', '100000'])
        contract = build_contract(args, self.path)
        source_contract = copy.deepcopy(contract)
        source_contract['settings']['render_steps'] = 25000
        source_contract['source_sha256']['train_fabric.py'] = (
            'e150e3518c433e8d68e65b0670a3766c6b1b9fd12a11c9adea432dacaf508661')
        model = cpu_model()
        model.feature_version = FEATURE_VERSION
        candidate_state = {'stage': 'map', 'step': 15, 'total_step': 15,
                           'baseline': None, 'refs': {}}
        metrics = {'map_mean': 0.2}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            old = output / 'best_map.step-00000015.pth'
            save_checkpoint(old, model, source_contract, candidate_state, metrics=metrics)
            ref = {'path': old.name, 'sha256': file_hash(old), 'step': 15}
            migrated_contract = copy.deepcopy(contract)
            migrated_contract['render_schedule_migration'] = {'kind': 'render_schedule_migration'}
            migrated = _rewrite_selection_candidate(output, ref, migrated_contract,
                                                    migration_tag='migrated-render100k')
            _cleanup_checkpoint_candidates(output)
            self.assertTrue((output / migrated['path']).is_file())
            self.assertFalse(old.exists())
            restored = cpu_model()
            restored.feature_version = FEATURE_VERSION
            loaded = load_checkpoint(output / migrated['path'], restored, migrated_contract)
            self.assertEqual(loaded['state']['total_step'], 15)

    def test_precision_changes_do_not_expand_historical_source_migration(self):
        current = build_contract(parse_args(['--out', 'unused']), self.path)
        historical = copy.deepcopy(current)
        historical['source_sha256']['train_fabric.py'] = (
            'e150e3518c433e8d68e65b0670a3766c6b1b9fd12a11c9adea432dacaf508661')
        historical['source_sha256']['model/nfplight_model.py'] = (
            'd5ea7cafe6449bbd0ab10dc2bb8074f139afe9ef6147449eed308408b828d0c5')
        with self.assertRaisesRegex(ValueError, 'source hash differs'):
            _assert_transition_contract(historical, current, allow_render_steps=True)

    def test_legacy_l1_policy_migration_is_explicit_and_narrow(self):
        current = build_contract(parse_args(['--out', 'unused']), self.path)
        legacy = copy.deepcopy(current)
        legacy['settings'].pop('checkpoint_every')
        legacy['settings'].pop('normal_loss')
        legacy.pop('normal_objective')
        legacy['source_sha256']['train_fabric.py'] = LEGACY_L1_TRAIN_SOURCE_SHA256
        model = cpu_model()
        model.feature_version = FEATURE_VERSION
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'legacy.pt'
            save_checkpoint(checkpoint, model, legacy, {'stage': 'map', 'step': 0, 'total_step': 0})
            strict = cpu_model()
            strict.feature_version = FEATURE_VERSION
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                load_checkpoint(checkpoint, strict, current)
            migrated = cpu_model()
            migrated.feature_version = FEATURE_VERSION
            load_checkpoint(checkpoint, migrated, current, allow_legacy_l1_resume_migration=True)
            cosine = build_contract(parse_args(['--out', 'unused', '--normal-loss', 'cosine']), self.path)
            rejected = cpu_model()
            rejected.feature_version = FEATURE_VERSION
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                load_checkpoint(checkpoint, rejected, cosine, allow_legacy_l1_resume_migration=True)

    def test_step_1000_commit_is_resumable(self):
        model = cpu_model()
        optimizer = torch.optim.Adam(model.net_g.parameters(), lr=1e-4)
        rng = random.Random(23)
        contract = {'manifest_sha256': file_hash(self.path),
                    'settings': {'checkpoint_every': 1000}}
        state = {'stage': 'map', 'step': 1000, 'total_step': 1000,
                 'refs': {}, 'history': []}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _commit_last(output, model, contract, state, optimizer, rng)
            checkpoint = output / 'last.pt'
            self.assertTrue(checkpoint.is_file())
            restored = load_checkpoint(checkpoint, cpu_model(), contract)
            self.assertEqual(restored['state']['total_step'], 1000)
            self.assertEqual(restored['state']['step'], 1000)

    def test_render_identity_invalid_lights_schedule_and_selection(self):
        model = cpu_model()
        target = FabricDataset(self.path, split='train')[0][None]
        lights = torch.tensor([[0., 0., 3.], [0.5, -0.5, 4.]])
        valid = torch.ones(1, 1, 256, 256, dtype=torch.bool)
        valid[..., :8, :8] = False
        self.assertEqual(float(fabric_render_loss(model, target, target, 2, lights, valid)), 0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            fabric_render_loss(model, target, target, 0)
        with self.assertRaisesRegex(ValueError, 'above'):
            fabric_render_loss(model, target, target, 2, -lights, valid)
        with self.assertRaisesRegex(ValueError, 'zero valid support'):
            fabric_render_loss(model, target, target, 2, lights, torch.zeros_like(valid))
        self.assertAlmostEqual(cosine_lr(0, 100, 5e-4), 5e-4)
        self.assertAlmostEqual(cosine_lr(99, 100, 5e-4), 1e-5)
        baseline = {f'map_{name}': 0.1 for name in ('normal', 'diffuse', 'roughness', 'specular')}
        candidate = {**baseline, 'render_l1': 0.01}
        self.assertTrue(render_candidate(candidate, baseline, 0.02, 0.02))
        candidate['map_roughness'] = 0.2
        self.assertFalse(render_candidate(candidate, baseline, 0.02, 0.02))

    def test_training_sampling_is_fixed_cached_tile_without_d4(self):
        dataset = FabricDataset(self.path, split='train')
        rng = random.Random(123)
        batch, slots = _sample_batch(dataset, rng, 4)
        self.assertTrue(all(slot[2:] == (0, False) for slot in slots))
        for item, (index, crop, _, _) in zip(batch, slots):
            torch.testing.assert_close(item, dataset.get_crop(index, crop))
        expected_rng = random.Random(123)
        expected_slots = [(expected_rng.randrange(len(dataset)),
                           expected_rng.randrange(dataset.manifest['crop_count']), 0, False)
                          for _ in range(4)]
        self.assertEqual(slots, expected_slots)
        self.assertEqual(rng.getstate(), expected_rng.getstate())

    def test_d4_sampling_rng_contract_and_transform_coverage(self):
        dataset = FabricDataset(self.path, split='train')

        class SequenceRng:
            def __init__(self):
                self.value = 0

            def randrange(self, stop):
                value = self.value % stop
                self.value += 1
                return value

        rng = random.Random(71)
        reference_rng = random.Random(71)
        transform_rng = SequenceRng()
        batch, slots = dataset.sample(
            rng, 8, augmentation='d4', augmentation_rng=transform_rng
        )
        expected_slots = [(reference_rng.randrange(len(dataset)),
                           reference_rng.randrange(dataset.manifest['crop_count']))
                          for _ in range(8)]
        self.assertEqual([(slot[0], slot[1]) for slot in slots], expected_slots)
        self.assertEqual(rng.getstate(), reference_rng.getstate())
        self.assertEqual(
            {slot[2:] for slot in slots},
            {(rotation, hflip) for rotation in range(4) for hflip in (False, True)},
        )
        for item, (index, crop, rotation, hflip) in zip(batch, slots):
            torch.testing.assert_close(
                item, dataset.get_crop(index, crop, rotation=rotation, hflip=hflip)
            )

        baseline = build_contract(parse_args(['--out', 'unused']), self.path)
        augmented = build_contract(
            parse_args(['--out', 'unused', '--augmentation', 'd4']), self.path
        )
        self.assertEqual(baseline['settings']['augmentation'], 'none')
        self.assertEqual(augmented['settings']['augmentation'], 'd4')
        self.assertEqual(augmented['input']['augmentation'], 'd4')
        self.assertEqual(augmented['data_policy']['augmentation'], 'd4')

    def test_invalid_label_corruption_does_not_change_masked_loss(self):
        generator = torch.Generator().manual_seed(2)
        target = torch.randn(1, 10, 4, 4, generator=generator).clamp(-1, 1)
        prediction = target.clone()
        valid = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        valid[:, :, :2] = True
        corrupted = target.clone()
        corrupted[:, :, 2:] = torch.randn_like(corrupted[:, :, 2:]) * 100
        first, first_parts = map_loss(prediction, target, valid)
        second, second_parts = map_loss(prediction, corrupted, valid)
        torch.testing.assert_close(first, second)
        for name in first_parts:
            torch.testing.assert_close(first_parts[name], second_parts[name])
        with self.assertRaisesRegex(ValueError, 'zero valid support'):
            map_loss(prediction, target, torch.zeros_like(valid))

    def test_candidate_and_generation_crash_are_recoverable(self):
        model = cpu_model()
        optimizer = torch.optim.Adam(model.net_g.parameters(), lr=1e-4)
        rng = random.Random(7)
        contract = {'manifest_sha256': file_hash(self.path)}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state = {'stage': 'map', 'step': 1, 'total_step': 1, 'refs': {}}
            _commit_last(output, model, contract, state, optimizer, rng)
            candidate_state = {'stage': 'map', 'step': 2, 'total_step': 2, 'refs': {}}
            candidate = _write_candidate(output, 'best_map', model, contract, candidate_state,
                                         {'map_mean': 0.1})
            self.assertTrue((output / candidate['path']).is_file())
            resumed = load_checkpoint(output / 'last.pt', cpu_model(), contract)
            self.assertEqual(resumed['state']['total_step'], 1)
            _cleanup_checkpoint_candidates(output)
            self.assertFalse((output / candidate['path']).exists())
            # A crash after immutable generation write leaves the committed last intact.
            orphan_state = {'stage': 'map', 'step': 3, 'total_step': 3, 'refs': {}}
            generation = output / 'last.step-00000003.map.pt'
            save_checkpoint(generation, model, contract, orphan_state, optimizer=optimizer, rng=rng)
            self.assertEqual(load_checkpoint(output / 'last.pt', cpu_model(), contract)['state']['total_step'], 1)
            _cleanup_checkpoint_candidates(output)
            self.assertFalse(generation.exists())


if __name__ == '__main__':
    unittest.main()
