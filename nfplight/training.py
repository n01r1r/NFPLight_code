from contextlib import contextmanager
import copy
import math
from pathlib import Path
import random

import torch

import numpy as np
import psutil

from nfplight.checkpoint import load_payload
from nfplight.storage import cpu_tree, save_checkpoint, write_json


def capture_rng():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(saved):
    random.setstate(saved['python'])
    state = saved['numpy']
    np.random.set_state((state[0], np.asarray(state[1], dtype = np.uint32), *state[2:]))
    torch.random.set_rng_state(saved['torch'])
    if saved['cuda']:
        if not torch.cuda.is_available() or len(saved['cuda']) != torch.cuda.device_count():
            raise ValueError('exact resume requires the original CUDA device population')
        torch.cuda.set_rng_state_all(saved['cuda'])


@contextmanager
def evaluation_mode(network):
    saved = capture_rng()
    modes = [(module, module.training) for module in network.modules()]
    network.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(saved)


def finite_metrics(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_metrics(item)
    elif isinstance(value, list):
        for item in value:
            finite_metrics(item)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(value) or value < 0:
            raise ValueError('metrics must be finite and nonnegative')


def batch_contract(config):
    batch = config['batch_size']
    micro = config.get('micro_batch_size', batch)
    accumulation = config.get('accumulation_steps', 1)
    if (any(type(value) is not int or value < 1 for value in (batch, micro, accumulation))
            or micro * accumulation != batch):
        raise ValueError('batch_size must equal micro_batch_size * accumulation_steps (positive integers)')
    return {'batch_size': batch, 'micro_batch_size': micro, 'accumulation_steps': accumulation}


class Trainer:
    def __init__(self, network, identity, objective, evaluator, config):
        self.net = network
        self.identity = copy.deepcopy(identity)
        self.policy = copy.deepcopy(identity['policy'])
        self.config = config
        self.objective, self.evaluator = objective, evaluator
        self.sample_count, self.batch_size = config['sample_count'], config['batch_size']
        self.batching = batch_contract(config)
        self._step_in_progress = False
        if not 1 <= self.batch_size <= self.sample_count:
            raise ValueError('batch size outside sample population')
        if self.policy.get('optimizer', 'Adam') != 'Adam' or self.policy.get('ema_decay') is not None:
            raise ValueError('continuation requires Adam without EMA')
        for key in ('evaluation_interval', 'checkpoint_interval', 'floor_observations'):
            if type(self.policy[key]) is not int or self.policy[key] < 1:
                raise ValueError(f'policy requires positive integer {key}')
        seed = self.policy['seed']
        self.samples = random.Random(seed + config['sampler_seed_offset'])
        self.d4 = random.Random(seed + config['d4_seed_offset'])
        self.augmentation_rng = random.Random(seed + config['augmentation_seed_offset'])
        self.noise_generator = torch.Generator(device = 'cpu').manual_seed(seed + config['noise_seed_offset'])
        self.state = {'seen_slots': [], 'stage': 'map', 'stage_step': 0, 'total_step': 0,
                      'map_updates': 0, 'render_updates': 0, 'status': 'running',
                      'stop_reasons': {}, 'best_map': None, 'best_render': None, 'history': [],
                      'evaluations': {'map': 0, 'render': 0},
                      'convergence': {stage: {'meaningful_best': None, 'floor_bad_observations': 0,
                                             'floor_reached': False} for stage in ('map', 'render')}}
        self.seen_slots = set()
        self.new_optimizer('map')

    def new_optimizer(self, stage):
        policy = self.policy
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = policy[stage + '_lr'],
                                          betas = tuple(self.config['adam_betas']),
                                          weight_decay = policy.get('weight_decay', 0.))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode = policy['mode'], factor = policy['factor'],
            patience = policy['patience'], threshold = policy['threshold'],
            threshold_mode = policy['threshold_mode'], cooldown = policy['cooldown'],
            min_lr = policy[stage + '_min_lr'], eps = policy['eps'])

    def snapshot(self):
        if self._step_in_progress:
            raise ValueError('checkpoint requires a completed optimizer update; restore after a failed step')
        self.state['seen_slots'] = sorted(self.seen_slots)
        return cpu_tree({'relation_version': 'spatial_inverse_square_v1', 'checkpoint_version': 1,
                         'package': 'full_dataset_suite', 'identity': self.identity, 'policy': self.policy,
                         'state': self.state, 'net_g': self.net.state_dict(),
                         'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                         'rng': capture_rng(), 'sample_rng': self.samples.getstate(),
                         'd4_rng': self.d4.getstate(), 'augmentation_rng': self.augmentation_rng.getstate(),
                         'noise_rng': self.noise_generator.get_state(), 'ema': None, 'batching': self.batching,
                         **getattr(self, 'checkpoint_extra', {})})

    def restore(self, payload):
        legacy_batching = batch_contract({'batch_size': self.batch_size})
        if payload.get('batching', legacy_batching) != self.batching:
            raise ValueError('checkpoint batching differs')
        required = {'relation_version', 'checkpoint_version', 'package', 'identity', 'policy', 'state',
                    'net_g', 'optimizer', 'scheduler', 'rng', 'sample_rng', 'd4_rng', 'augmentation_rng', 'noise_rng', 'ema'}
        if not required <= payload.keys():
            raise ValueError('model-only checkpoint cannot resume training')
        if (payload['relation_version'] != 'spatial_inverse_square_v1'
                or payload['checkpoint_version'] != 1 or payload['package'] != 'full_dataset_suite'
                or payload['identity'] != self.identity or payload['policy'] != self.policy or payload['ema'] is not None):
            raise ValueError('checkpoint contract differs; legacy relation optimizer resume is forbidden')
        stage = payload['state']['stage']
        if stage not in ('map', 'render'):
            raise ValueError('invalid checkpoint phase')
        for key in ('stage_step', 'total_step', 'map_updates', 'render_updates'):
            value = payload['state'][key]
            if type(value) is not int or value < 0:
                raise ValueError(f'invalid checkpoint step counter: {key}')
        if payload['state']['total_step'] != payload['state']['map_updates'] + payload['state']['render_updates']:
            raise ValueError('checkpoint update counters disagree')
        finite_metrics(payload['state'])
        for group in payload['optimizer']['param_groups']:
            for key in ('lr', 'eps', 'weight_decay'):
                if not math.isfinite(group[key]) or group[key] < 0 or (key == 'eps' and group[key] == 0):
                    raise ValueError(f'invalid optimizer setting: {key}')
            if any(not math.isfinite(value) or not 0 <= value < 1 for value in group['betas']):
                raise ValueError('invalid optimizer betas')
        scheduler = payload['scheduler']
        if math.isnan(scheduler['best']) or any(not math.isfinite(value) or value < 0 for value in scheduler['_last_lr']):
            raise ValueError('invalid scheduler state')
        for value in payload['net_g'].values():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('invalid checkpoint model tensor')
        for state in payload['optimizer']['state'].values():
            for value in state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError('nonfinite optimizer state')
        self.net.load_state_dict(payload['net_g'], strict = True)
        self.new_optimizer(stage)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.scheduler.load_state_dict(payload['scheduler'])
        self.samples.setstate(payload['sample_rng'])
        self.d4.setstate(payload['d4_rng'])
        self.augmentation_rng.setstate(payload['augmentation_rng'])
        self.noise_generator.set_state(payload['noise_rng'])
        self.state = copy.deepcopy(payload['state'])
        self.seen_slots = set(self.state['seen_slots'])
        restore_rng(payload['rng'])
        self._step_in_progress = False

    def step(self):
        if self._step_in_progress:
            raise ValueError('restore checkpoint before retrying a failed optimizer update')
        self._step_in_progress = True
        self.net.train()
        indices = self.samples.sample(range(self.sample_count), self.batch_size)
        transforms = [self.d4.randrange(8) for _ in indices]
        self.optimizer.zero_grad(set_to_none = True)
        micro = self.batching['micro_batch_size']
        accumulated = self.batching['accumulation_steps'] > 1
        total_weight, total_loss, metrics = 0., 0., {}
        pixel_scale = None
        for start in range(0, self.batch_size, micro):
            loss, parts = self.objective(self.net, indices[start:start + micro],
                transforms[start:start + micro], self.state['stage'])
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError('objective must be finite scalar')
            finite_metrics(parts)
            weight = 1.
            if accumulated:
                pixels = parts.get('valid_pixels')
                if type(pixels) is not int or pixels <= 0:
                    raise ValueError('accumulation requires positive integer valid_pixels from objective')
                pixel_scale = pixels if pixel_scale is None else pixel_scale
                weight = pixels / pixel_scale
            (loss * weight if accumulated else loss).backward()
            total_weight += weight
            total_loss += float(loss.detach()) * weight
            for key, value in parts.items():
                metrics[key] = metrics.get(key, 0.) + value * (1 if key == 'valid_pixels' else weight)
            del loss
        if accumulated:
            for parameter in self.net.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(total_weight)
        metrics = {key: value if key == 'valid_pixels' else value / total_weight for key, value in metrics.items()}
        gradient = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config['clip_norm'], error_if_nonfinite = True)
        if not torch.isfinite(gradient) or gradient <= 0:
            raise ValueError('gradient must be finite and nonzero')
        self.optimizer.step()
        self.seen_slots.update(indices)
        self.state['stage_step'] += 1
        self.state['total_step'] += 1
        self.state[self.state['stage'] + '_updates'] += 1
        self._step_in_progress = False
        return {'stage': self.state['stage'], 'step': self.state['stage_step'],
                'total_step': self.state['total_step'], 'indices': indices, 'd4_transforms': transforms,
                'loss': total_loss / total_weight, 'gradient_norm': float(gradient),
                'lr': self.optimizer.param_groups[0]['lr'], **metrics}

    def evaluate(self):
        stage = self.state['stage']
        with evaluation_mode(self.net):
            populations = self.evaluator(self.net, stage, self.state['stage_step'])
        finite_metrics(populations)
        mean = populations['heldout']['mean']
        maps, render = float(mean['ndrs_sum']), float(mean['render_l1'])
        metric = maps if stage == 'map' else render
        eligible = stage == 'map' or maps <= self.policy['render_ndr_guard'] * self.state['best_map']['ndrs_sum']
        key = 'best_' + stage
        best = self.state[key]
        if eligible and (best is None or metric < best['metric']):
            self.state[key] = {'net_g': cpu_tree(self.net.state_dict()), 'metric': metric,
                               'ndrs_sum': maps, 'render_l1': render, 'stage': stage,
                               'step': self.state['stage_step'], 'total_step': self.state['total_step'],
                               'evaluation_weights': 'online'}
        before = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(metric)
        after = self.optimizer.param_groups[0]['lr']
        record = self.state['convergence'][stage]
        improved = record['meaningful_best'] is None or metric < record['meaningful_best'] * (1 - self.policy['threshold'])
        if improved:
            record['meaningful_best'] = metric
            record['floor_bad_observations'] = 0
        floor = self.policy[stage + '_min_lr']
        at_floor = after <= floor + 1e-15
        if not at_floor or not record['floor_reached']:
            record['floor_bad_observations'] = 0
        elif before <= floor + 1e-15 and not improved:
            record['floor_bad_observations'] += 1
        record['floor_reached'] = at_floor
        self.state['evaluations'][stage] += 1
        self.state['history'].append({'stage': stage, 'step': self.state['stage_step'],
            'total_step': self.state['total_step'], 'populations': populations, 'evaluation_weights': 'online',
            'eligible_for_selection': eligible, 'monitor': metric, 'meaningful_improvement': improved,
            'lr_before': before, 'lr': after,
            'scheduler': {key: copy.deepcopy(value) for key, value in self.scheduler.state_dict().items() if key != 'mode_worse'},
            'convergence': copy.deepcopy(record)})
        return record['floor_bad_observations'] >= self.policy['floor_observations']

    def save(self, output):
        output = Path(output)
        save_checkpoint(output / 'last.pt', self.snapshot(), reserve_bytes = self.config['disk_reserve_bytes'])
        write_json(output / 'status.json', {key: self.state[key] for key in
                   ('status', 'stage', 'stage_step', 'total_step', 'stop_reasons', 'evaluations', 'convergence')})
        write_json(output / 'evaluation.json', self.state['history'])

    def run(self, output, caps, *, max_updates = None):
        output = Path(output)
        if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
            raise ValueError('smoke update limit must be a positive integer')
        if self.state['status'] in ('complete', 'screening_complete'):
            raise ValueError('completed runs cannot be silently restarted')
        if not self.state['history']:
            self.evaluate()
        self.state['status'] = 'running'
        self.save(output)
        updates = 0
        while True:
            stage = self.state['stage']
            cap = caps[stage + '_cap']
            if cap is not None and self.state['stage_step'] >= cap:
                self.state['status'] = 'budget_paused'
                self.save(output)
                return self.state
            if psutil.virtual_memory().available < self.config.get('ram_reserve_bytes', 0):
                self.restore(load_payload(output / 'last.pt'))
                self.state['status'] = 'memory_paused'
                self.save(output)
                return self.state
            row = self.step()
            write_json(output / 'last_update.json', row)
            updates += 1
            stopped = self.evaluate() if self.state['stage_step'] % self.policy['evaluation_interval'] == 0 else False
            if stopped:
                self.state['stop_reasons'][stage] = 'floor_no_meaningful_improvement'
                if stage == 'map':
                    save_checkpoint(output / 'map_stage_end.pt', self.snapshot(), reserve_bytes = self.config['disk_reserve_bytes'])
                    self.net.load_state_dict(self.state['best_map']['net_g'], strict = True)
                    self.state.update(stage = 'render', stage_step = 0)
                    self.state['render_start_sample_offset'] = self.state['map_updates']
                    self.new_optimizer('render')
                    self.evaluate()
                else:
                    self.state['status'] = 'complete'
                    self.state['selected'] = 'best_render' if self.state['best_render'] is not None else 'best_map'
                    self.save(output)
                    selected = self.state[self.state['selected']]
                    save_checkpoint(output / 'selected.pt', {'net_g': selected['net_g'], 'identity': self.identity,
                        'relation_version': 'spatial_inverse_square_v1'}, reserve_bytes = self.config['disk_reserve_bytes'])
                    return self.state
            if max_updates is not None and updates >= max_updates:
                self.state['status'] = 'interrupted'
                self.save(output)
                return self.state
            if stopped or self.state['stage_step'] % self.policy['checkpoint_interval'] == 0:
                self.save(output)
