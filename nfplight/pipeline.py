import time

import torch

from einops import repeat
import numpy as np

from nfplight.checkpoint import decode
from nfplight.data import MaterialBank, canonical_target, fixed_f0, map_loss
from nfplight.features import Features
from nfplight.rendering import Renderer, lights_from_config


class TrainingPipeline:
    def __init__(self, execution, profile, runtime, *, device):
        self.execution, self.profile, self.runtime = execution, profile, runtime
        loss = execution['scientific_settings']['loss']
        if (loss['map_terms'] != ['N', 'D', 'R', 'S'] or loss['map_aggregation'] != 'sum'
                or loss['normal'] != 'unit XYZ L1' or profile['decoder'] != 'unit_xyz'):
            raise ValueError('training requires unit XYZ and NDRS component L1 sum')
        dataset = execution['scientific_settings'].get('dataset', {})
        if (dataset.get('validation_crop_indices', runtime['validation_crops']) != runtime['validation_crops']
                or dataset.get('train_crop_indices', list(range(runtime['crop_count']))) != list(range(runtime['crop_count']))):
            raise ValueError('crop populations differ from execution identity')
        self.device = torch.device(device)
        self.features = Features(profile['features'], device = device)
        self.renderer = Renderer(runtime['renderer'], device = device)
        self.f0 = execution['scientific_settings']['rendering']['f0']
        if not 0 <= self.f0 <= 1 or loss['specular_target'] != self.f0:
            raise ValueError('specular supervision and rendering F0 disagree')
        self.crop_count = runtime['crop_count']
        self.train = MaterialBank(execution['manifest'], 'train', cache_size = runtime['cache_size'], f0 = self.f0, crop_count = self.crop_count)
        self.val = MaterialBank(execution['manifest'], 'val', cache_size = runtime['cache_size'], f0 = self.f0, crop_count = self.crop_count)
        if self.train.records != execution['train'] or self.val.records != execution['validation']:
            raise ValueError('manifest population differs from checkpoint execution identity')
        train_groups = {row['group_id'] for row in self.train.records}
        if train_groups & {row['group_id'] for row in self.val.records}:
            raise ValueError('train/validation source groups overlap')
        rendering = execution['scientific_settings']['rendering']
        if (rendering.get('precision') != 'float32' or rendering.get('amp') is not False
                or rendering.get('tf32') is not False
                or rendering['relation_version'] != profile['features']['relation']
                or rendering['feature_version'] != profile['feature_version']
                or any(rendering[key] != profile['features'][key] for key in ('near', 'far'))
                or rendering['gain'] != profile['features']['training_gain']):
            raise ValueError('training feature contract differs from config')
        self.training_lights = lights_from_config(rendering['training_lights'], device = device)
        self.validation_lights = lights_from_config(rendering['novel_lights'], device = device)

    def prepare(self, target):
        target, mask = canonical_target(target.to(self.device), roughness_floor = self.runtime['renderer']['roughness_floor'])
        target = fixed_f0(target, self.f0)
        with torch.no_grad():
            cfg = self.profile['features']
            pair = self.renderer.observations(target, cfg['near'], cfg['far']) * mask
            features, _ = self.features(pair, purpose = 'synthetic', mask = mask)
        return target, mask, features

    def render_loss(self, pred, target, mask, lights, lighting):
        views = repeat(target.new_tensor(lighting['view']), 'c -> l c', l = len(lights))
        with torch.no_grad():
            reference = self.renderer(target, lights, views)
        rendered = self.renderer(fixed_f0(pred, self.f0), lights, views)
        return (rendered - reference).abs().masked_select(mask[:, None].expand_as(rendered)).mean()

    def objective(self, network, indices, transforms, stage):
        target = torch.stack([self.train.crop(index // self.crop_count, index % self.crop_count, transform % 4, transform >= 4)
                              for index, transform in zip(indices, transforms)])
        target, mask, features = self.prepare(target)
        pred = decode(network(features), self.profile['decoder'])
        maps, parts = map_loss(pred, target, mask)
        settings = self.execution['scientific_settings']
        rendered = self.render_loss(pred, target, mask, self.training_lights, settings['rendering']['training_lights']) if stage == 'render' else maps.new_zeros(())
        weight = settings['loss']['render_weight']
        if not 0 < weight < float('inf'):
            raise ValueError('render weight must be finite and positive')
        return maps + weight * rendered, {'ndrs_sum': float(maps.detach()), 'render_loss': float(rendered.detach()),
            **dict(zip(('normal_loss', 'diffuse_loss', 'roughness_loss', 'specular_loss'), [float(part.detach()) for part in parts]))}

    def evaluate(self, network, stage, step):
        started = time.perf_counter()
        rows = []
        for index, record in enumerate(self.val.records):
            crops = []
            for crop in self.runtime['validation_crops']:
                target, mask, features = self.prepare(self.val.crop(index, crop)[None])
                pred = decode(network(features), self.profile['decoder'])
                _, parts = map_loss(pred, target, mask)
                rendered = self.render_loss(pred, target, mask, self.validation_lights,
                    self.execution['scientific_settings']['rendering']['novel_lights'])
                angle = torch.rad2deg(torch.acos((pred[:, :3] * target[:, :3]).sum(1, keepdim = True).clamp(-1, 1))).masked_select(mask).mean()
                values = [float(part) for part in parts] + [float(rendered), float(angle)]
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite validation')
                crops.append(values)
            values = np.mean(crops, axis = 0)
            rows.append({**{key: record[key] for key in ('material_id', 'category', 'group_id')},
                **dict(zip(('map_normal', 'map_diffuse', 'map_roughness', 'map_specular', 'render_l1', 'normal_degrees'), map(float, values))),
                'ndr_mean': float(np.mean(values[:3])), 'ndrs_sum': float(sum(values[:4]))})
        keys = ('map_normal', 'map_diffuse', 'map_roughness', 'map_specular', 'render_l1', 'normal_degrees', 'ndr_mean', 'ndrs_sum')
        def mean(subset):
            return {key: float(np.mean([row[key] for row in subset])) for key in keys}
        groups = [mean([row for row in rows if row['group_id'] == group]) for group in sorted({row['group_id'] for row in rows})]
        return {'heldout': {'mean': mean(rows), 'materials': rows,
            'category': {category: {'materials': sum(row['category'] == category for row in rows),
                **mean([row for row in rows if row['category'] == category])} for category in sorted({row['category'] for row in rows})},
            'source_group_macro': {key: float(np.mean([group[key] for group in groups])) for key in keys},
            'sample_folders': len(rows), 'source_groups': len(groups), 'seconds': time.perf_counter() - started,
            'crop_indices': self.runtime['validation_crops'], 'novel_lights': len(self.validation_lights), 'official_test_payloads_read': 0}}
