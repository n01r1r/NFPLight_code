"""Deterministic CPU regression tests for Fabric contracts and resume behavior."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from data.dataset import RealDataset
from data.fabric import FabricDataset, decode_tile, file_hash, prepare, write_json
from model.nfplight_model import NFPLightModel, checkpoint_feature_version
from train_fabric import (FEATURE_VERSION, LEGACY_L1_TRAIN_SOURCE_SHA256, build_contract, cosine_lr, evaluate, load_checkpoint, optimizer_step,
                          render_candidate, save_checkpoint, _cleanup_checkpoint_candidates,
                          _sample_batch, _write_candidate, _commit_last, _checkpoint_due,
                          map_loss, parse_args)
from train_scratch import render_loss
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

    def test_renderer_png_real_loader_feature_roundtrip(self):
        model = cpu_model()
        target = FabricDataset(self.path, split='train').get_crop(0, 0)[None]
        rendered = model.render_input_images(target, toLDR=False)
        expected = model.render_input_images(target, toLDR=True)
        encoded = (rendered.pow(1 / 2.2) * 255 + 0.5).clamp(0, 255).to(torch.uint8)[0]
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / 'capture'
            capture.mkdir()
            for name, pixels in zip(['near', 'far'], encoded.chunk(2)):
                Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(capture / f'{name}.png')
            dataset = RealDataset({'root': temporary, 'imageSize': 256, 'input_gamma': True})
            actual = dataset[0]['inputs'][None]
            torch.testing.assert_close(expected, actual, rtol=0, atol=1e-7)
            torch.testing.assert_close(model.build_features(expected)[0], model.build_features(actual)[0], rtol=0, atol=1e-6)

    def test_fixed_evaluation_preserves_training_rng_and_mode(self):
        torch.manual_seed(11)
        model = cpu_model()
        dataset = FabricDataset(self.path, split='val')
        before = torch.get_rng_state().clone()
        first = evaluate(model, dataset, crop_count=2, light_count=2)
        second = evaluate(model, dataset, crop_count=2, light_count=2)
        self.assertEqual(first, second)
        self.assertTrue(model.net_g.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(row['crops'] == 2 for row in first['materials']))

    def test_optimizer_checkpoint_exact_next_step(self):
        torch.manual_seed(4)
        model = cpu_model()
        dataset = FabricDataset(self.path, split='train')
        optimizer = torch.optim.Adam(model.net_g.parameters(), lr=1e-4)
        rng = random.Random(17)
        options = dict(batch_size=1, accumulate=2, weight=0.5, light_count=2)
        optimizer_step(model, dataset, optimizer, rng, **options)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'resume.pt'
            contract = {'manifest_sha256': file_hash(self.path)}
            save_checkpoint(path, model, contract,
                            {'stage': 'render', 'step': 1000, 'total_step': 1000},
                            optimizer=optimizer, rng=rng)
            expected = optimizer_step(model, dataset, optimizer, rng, **options)
            restored = cpu_model()
            restored_optimizer = torch.optim.Adam(restored.net_g.parameters(), lr=1e-4)
            restored_rng = random.Random(99)
            load_checkpoint(path, restored, contract, optimizer=restored_optimizer, rng=restored_rng)
            actual = optimizer_step(restored, dataset, restored_optimizer, restored_rng, **options)
            self.assertEqual(expected, actual)
            for a, b in zip(model.net_g.parameters(), restored.net_g.parameters()):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                load_checkpoint(path, restored, {'manifest_sha256': 'changed'})

    def test_checkpoint_cadence_default_and_step_1000_resume_boundary(self):
        self.assertEqual(parse_args(['--out', 'unused']).checkpoint_every, 1000)
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
        self.assertEqual(float(render_loss(model, target, target, 2, lights)), 0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            render_loss(model, target, target, 0)
        with self.assertRaisesRegex(ValueError, 'above'):
            render_loss(model, target, target, 2, -lights)
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
