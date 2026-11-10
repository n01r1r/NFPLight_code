import time

import torch

from einops import repeat
import numpy as np

from nfplight.checkpoint import decode
from nfplight.data import MaterialBank, canonical_target, fixed_f0, map_loss
from nfplight.features import Features
from nfplight.rendering import Renderer, lights_from_config


def validation_metrics(pred, target, mask, rendered, reference):
    if (pred.shape != target.shape or pred.ndim != 4 or len(pred) < 1 or pred.shape[1] != 10
            or pred.dtype != torch.float32 or target.dtype != torch.float32
            or mask.dtype != torch.bool or mask.shape != (len(pred), 1, *pred.shape[-2:])
            or not mask.flatten(1).any(1).all()
            or rendered.shape != reference.shape or rendered.ndim != 5
            or rendered.shape[0] != len(pred) or rendered.shape[2:] != (3, *pred.shape[-2:])
            or rendered.shape[1] < 1 or rendered.dtype != torch.float32 or reference.dtype != torch.float32
            or not torch.isfinite(pred).all() or not torch.isfinite(target).all()
            or not torch.isfinite(rendered).all() or not torch.isfinite(reference).all()):
        raise ValueError('invalid FP32 validation inputs/support')
    pixels = mask.flatten(1).sum(1)
    parts = []
    for section in (slice(0, 3), slice(3, 6), slice(6, 7), slice(7, 10)):
        delta = (pred[:, section] - target[:, section]).abs()
        parts.append(delta.masked_fill(~mask, 0).flatten(1).sum(1) / (pixels * delta.shape[1]))
    delta = (rendered - reference).abs()
    rendered_loss = delta.masked_fill(~mask[:, None], 0).flatten(1).sum(1) / (pixels * delta.shape[1] * 3)
    angle = torch.rad2deg(torch.acos((pred[:, :3] * target[:, :3]).sum(1, keepdim = True).clamp(-1, 1)))
    angle = angle.masked_fill(~mask, 0).flatten(1).sum(1) / pixels
    values = torch.stack([*parts, rendered_loss, angle], dim = 1)
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite validation')
    return values


class TrainingPipeline:
    def __init__(self, execution, profile, runtime, *, device):
        self.execution, self.profile, self.runtime = execution, profile, runtime
        self.validation_batch_size = runtime.get('validation_batch_size', 1)
        if type(self.validation_batch_size) is not int or self.validation_batch_size < 1:
            raise ValueError('validation_batch_size must be a positive integer')
        if self.validation_batch_size > 1 and profile['features']['normalization'] != 'sample':
            raise ValueError('batched validation requires sample feature normalization')
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
        if rendering.get('observation_quantization', 'none') != profile['features'].get('synthetic_quantization', 'none'):
            raise ValueError('synthetic observation quantization differs from config')
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
        return maps + weight * rendered, {'valid_pixels': int(mask.sum()),
            'ndrs_sum': float(maps.detach()), 'render_loss': float(rendered.detach()),
            **dict(zip(('normal_loss', 'diffuse_loss', 'roughness_loss', 'specular_loss'), [float(part.detach()) for part in parts]))}

    @torch.no_grad()
    def validation_batch(self, network, targets):
        target, mask, features = self.prepare(targets)
        pred = decode(network(features), self.profile['decoder'])
        lighting = self.execution['scientific_settings']['rendering']['novel_lights']
        views = repeat(target.new_tensor(lighting['view']), 'c -> l c', l = len(self.validation_lights))
        reference = self.renderer(target, self.validation_lights, views)
        rendered = self.renderer(fixed_f0(pred, self.f0), self.validation_lights, views)
        return validation_metrics(pred, target, mask, rendered, reference).cpu().numpy().astype(np.float64)

    def evaluate(self, network, stage, step):
        started = time.perf_counter()
        rows = []
        progress = getattr(self, 'progress', None)
        samples = [(index, crop) for index in range(len(self.val.records)) for crop in self.runtime['validation_crops']]
        total, completed = len(samples), 0
        requested = self.validation_batch_size
        batch_size = getattr(self, 'validation_batch_limit', requested)
        retries, used = 0, set()
        material_crops = [[] for _ in self.val.records]
        if progress:
            progress('validation', 0, total, stage = stage, stage_step = step,
                     batch_size = batch_size, requested_batch_size = requested, oom_retries = retries)
        while completed < total:
            size = min(batch_size, total - completed)
            batch = samples[completed:completed + size]
            targets = torch.stack([self.val.crop(index, crop) for index, crop in batch])
            try:
                values = self.validation_batch(network, targets)
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                values = None
            if values is None:
                batch_size = max(1, size // 2)
                retries += 1
                # Release the failed call's traceback/tensors before emptying the CUDA cache.
                torch.cuda.empty_cache()
                if progress:
                    progress('validation', completed, total, stage = stage, stage_step = step,
                             batch_size = batch_size, requested_batch_size = requested, oom_retries = retries)
                continue
            if values.shape != (size, 6) or not np.isfinite(values).all():
                raise ValueError('invalid per-crop validation metrics')
            for (index, crop), value in zip(batch, values):
                material_crops[index].append(value)
            completed += size
            used.add(size)
            if progress:
                progress('validation', completed, total, stage = stage, stage_step = step,
                         batch_size = size, requested_batch_size = requested, oom_retries = retries)
        self.validation_batch_limit = batch_size
        for index, record in enumerate(self.val.records):
            values = np.mean(material_crops[index], axis = 0)
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
            'crop_indices': self.runtime['validation_crops'], 'novel_lights': len(self.validation_lights), 'official_test_payloads_read': 0,
            'validation_batch_size_requested': requested, 'validation_batch_size_final': batch_size,
            'validation_batch_sizes_used': sorted(used), 'validation_oom_retries': retries}}
