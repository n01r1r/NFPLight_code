import argparse
import copy
import json
import platform
from pathlib import Path
import random

import torch

import numpy as np
import psutil

from nfplight.checkpoint import read_config, load_payload, load_model, check_profile, decode
from nfplight.features import Features
from nfplight.network import NFPLight
from nfplight.pipeline import TrainingPipeline
from nfplight.storage import identity, output_lock, save_checkpoint, write_json
from nfplight.training import Trainer

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def read_json(path):
    return json.loads(Path(path).read_text(encoding = 'utf-8-sig'))


def configure_runtime(runtime, device):
    torch.set_num_threads(runtime['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')


def infer(args):
    output = Path(args.out)
    if output.exists():
        raise FileExistsError('prediction output already exists')
    config = read_config(args.config or CONFIGS / '7m_spatial.json')
    runtime = read_json(args.runtime)
    configure_runtime(runtime, args.device)
    source = load_payload(args.input)
    if source.get('schema') != 'nfplight.input.v1' or source.get('capture_gain_mode') != 'author_patch20':
        raise ValueError('expected reviewed FP32 input with author_patch20 contract')
    if args.config is None:
        checkpoint = load_payload(args.checkpoint)
        rendering = checkpoint.get('identity', {}).get('scientific_settings', {}).get('rendering', {})
        if (checkpoint.get('feature_version', rendering.get('feature_version')) is None
                or checkpoint.get('relation_version', rendering.get('relation_version')) is None):
            raise ValueError('checkpoint lacks feature/relation metadata; specify an explicit --config')
        del checkpoint
    model = load_model(args.checkpoint, config, device = args.device)
    features = Features(config['features'], device = args.device)
    pair = source['pair'].to(args.device)
    mask = source['valid_mask'].to(args.device)
    with torch.inference_mode():
        inputs, trace = features(pair, purpose = 'capture', mask = mask)
        raw = model(inputs)
        prediction = decode(raw, config['decoder'])
    save_checkpoint(output, {'schema': 'nfplight.prediction.v1', 'config': config,
        'input_identity': identity(args.input), 'input_spatial_policy': source.get('spatial_policy', 'external_calibrated_pair'),
        'checkpoint_identity': identity(args.checkpoint),
        'features': inputs, 'raw_prediction': raw, 'prediction': prediction, 'trace': trace,
        'channel_order': 'normal XYZ, diffuse RGB, roughness, specular RGB'}, reserve_bytes = runtime['disk_reserve_bytes'])


def execution_from_config(args, profile, runtime):
    execution = copy.deepcopy(read_json(args.training_config))
    if not args.manifest:
        raise ValueError('fresh training requires --manifest')
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding = 'utf-8').splitlines() if line.strip()]
    train = [row for row in rows if row['split'] == 'train']
    val = [row for row in rows if row['split'] == 'val']
    if not train or not val or len({row['material_id'] for row in train + val}) != len(train + val):
        raise ValueError('empty or duplicate train/validation population')
    net = profile['network']
    execution['scientific_settings']['backbone'] = {'width': net['dim'], 'encoder_blocks': net['encoder_depths'],
        'middle_blocks': net['middle_depth'], 'decoder_blocks': net['decoder_depths']}
    execution['policy']['dropout_probability'] = net['dropout']
    execution.update(manifest = str(Path(args.manifest).resolve()), train = train, validation = val,
        purpose = 'smoke' if args.smoke_updates else 'training', package = 'nfplight.clean.v1')
    execution['file_identities'] = [identity(args.manifest)] + [identity(row['shard_path']) for row in train + val]
    execution['scientific_settings']['dataset'] = {'train_crop_indices': list(range(runtime['crop_count'])),
        'validation_crop_indices': runtime['validation_crops'], 'train_slots': len(train) * runtime['crop_count']}
    return execution


def validate_resume(payload, execution, profile, runtime, *, device):
    check_profile(payload, profile)
    frozen_profile = payload.get('clean_profile')
    if frozen_profile is None:
        frozen_profile = read_config(CONFIGS / '7m_spatial.json')
    for key in ('network', 'features', 'decoder', 'feature_version'):
        if profile[key] != frozen_profile[key]:
            raise ValueError(f'exact resume profile mismatch: {key}')
    frozen_runtime = payload.get('clean_runtime')
    if frozen_runtime is None:
        frozen_runtime = read_json(CONFIGS / 'legacy_resume.json')
    for key in ('batch_size', 'adam_betas', 'clip_norm', 'sampler_seed_offset', 'd4_seed_offset',
                'augmentation_seed_offset', 'noise_seed_offset', 'crop_count', 'validation_crops', 'renderer', 'cpu_threads'):
        if runtime[key] != frozen_runtime[key]:
            raise ValueError(f'exact resume runtime mismatch: {key}')
    if execution.get('purpose') != 'training':
        raise ValueError('smoke checkpoints cannot become research training')
    if execution['policy'].get('dropout_probability') != profile['network']['dropout']:
        raise ValueError('dropout differs from checkpoint')
    if runtime['sample_count'] != len(execution['train']) * runtime['crop_count']:
        raise ValueError('sampler population differs')
    if payload.get('rng', {}).get('cuda') and not str(device).startswith('cuda'):
        raise ValueError('CUDA checkpoint exact resume cannot migrate to CPU')
    input_paths = {str(Path(execution['manifest']).resolve()),
                   *(str(Path(row['shard_path']).resolve()) for row in execution['train'] + execution['validation'])}
    recorded = {str(Path(item['path']).resolve()): item for item in execution.get('file_identities', [])}
    for path in input_paths:
        if path not in recorded or identity(path) != recorded[path]:
            raise ValueError(f'resume input identity changed or missing: {path}')


def runtime_environment(device):
    cuda = str(device).startswith('cuda')
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
            'device_name': torch.cuda.get_device_name(device) if cuda else 'cpu',
            'device_capability': list(torch.cuda.get_device_capability(device)) if cuda else None,
            'precision': 'float32', 'amp': False, 'tf32': False}


def train(args):
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError('training/migration requires a new output directory')
    profile, runtime = read_config(args.config or CONFIGS / '7m_spatial.json'), read_json(args.runtime)
    configure_runtime(runtime, args.device)
    if profile['features']['relation'] != 'spatial_inverse_square_v1':
        raise ValueError('training requires the current spatial relation contract')
    environment = runtime_environment(args.device)
    if psutil.virtual_memory().available < runtime['ram_reserve_bytes']:
        raise ValueError('RAM reserve unavailable')
    payload = load_payload(args.resume) if args.resume else None
    if payload is not None:
        if payload.get('execution_purpose') == 'smoke':
            raise ValueError('smoke continuation cannot become research training')
        recorded_environment = payload.get('runtime_environment')
        if recorded_environment is None:
            recorded_environment = read_json(Path(args.resume).parent / 'runtime.json')['environment']
        if recorded_environment != environment:
            raise ValueError('exact continuation requires the recorded Python/PyTorch/CUDA/device environment')
        execution = payload['identity']
        validate_resume(payload, execution, profile, runtime, device = args.device)
    else:
        execution = execution_from_config(args, profile, runtime)
    random.seed(execution['policy']['seed'])
    np.random.seed(execution['policy']['seed'])
    torch.manual_seed(execution['policy']['seed'])
    if str(args.device).startswith('cuda'):
        torch.cuda.manual_seed_all(execution['policy']['seed'])
    pipeline = TrainingPipeline(execution, profile, runtime, device = args.device)
    network = NFPLight(**profile['network']).to(args.device)
    count = sum(parameter.numel() for parameter in network.parameters())
    if payload is not None and count != execution['policy']['parameters']:
        raise ValueError('parameter count differs from resumable checkpoint')
    if payload is None:
        execution['policy']['parameters'] = count
    if runtime['sample_count'] != len(pipeline.train.records) * runtime['crop_count']:
        raise ValueError('runtime sample_count differs from manifest')
    trainer = Trainer(network, execution, pipeline.objective, pipeline.evaluate, runtime)
    trainer.checkpoint_extra = {'clean_runtime': runtime, 'clean_profile': profile, 'runtime_environment': environment,
        'execution_purpose': 'smoke' if args.smoke_updates else 'training'}
    if payload is not None:
        trainer.restore(payload)
    caps = {key: execution['policy'][key] for key in ('map_cap', 'render_cap')}
    if 'continuation' in execution:
        caps['render_cap'] = execution['continuation']['effective_render_cap']
    with output_lock(output):
        write_json(output / 'execution_config.json', execution)
        write_json(output / 'implementation.json', {'schema': 'nfplight.implementation.v1',
            'source_files': [identity(path) for path in sorted(Path(__file__).parent.glob('*.py'))],
            'profile': profile, 'runtime': runtime,
            'origin_checkpoint': None if payload is None else identity(args.resume),
            'purpose': 'smoke' if args.smoke_updates else 'training',
            'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'device': args.device, 'environment': environment, 'historical_identity_preserved': payload is not None,
            'sha256_computed': False})
        try:
            trainer.run(output, caps, max_updates = args.smoke_updates)
        except BaseException as error:
            write_json(output / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                'last_checkpoint_preserved': (output / 'last.pt').is_file()})
            raise


def main():
    parser = argparse.ArgumentParser(description = 'NFPLight FP32 inference and training')
    commands = parser.add_subparsers(dest = 'command', required = True)
    for name in ('infer', 'train'):
        command = commands.add_parser(name)
        command.add_argument('--config', help = 'model profile; default is 7m_spatial for identified checkpoints')
        command.add_argument('--runtime', default = str(CONFIGS / 'runtime.json'))
        command.add_argument('--device', default = 'cpu' if name == 'infer' else 'cuda')
        command.add_argument('--out', required = True)
        if name == 'infer':
            command.add_argument('--input', required = True)
            command.add_argument('--checkpoint', required = True)
        else:
            command.add_argument('--resume')
            command.add_argument('--manifest')
            command.add_argument('--training-config', default = str(CONFIGS / 'training.json'))
            command.add_argument('--smoke-updates', type = int)
    args = parser.parse_args()
    infer(args) if args.command == 'infer' else train(args)
