"""Explicit fresh/strict-resume Colab entry point for the unchanged frozen CLI."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from functools import partial
import shutil
import subprocess
import sys
import time

import psutil
import torch

from colab_tools import relative_path
from nfplight import cli, training
from nfplight.checkpoint import read_config
from nfplight.storage import atomic_write, identity, write_json


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def save_cpu_checkpoint(path, payload, *, reserve_bytes, backup_dir = None, progress = None):
    # snapshot() already owns a detached CPU tree; preserve the verified writer behavior.
    def cpu_only(value):
        if torch.is_tensor(value):
            require(value.device.type == 'cpu', 'checkpoint snapshot must be CPU-owned')
        elif isinstance(value, dict):
            for item in value.values():
                cpu_only(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                cpu_only(item)
    cpu_only(payload)
    if progress:
        progress('saving checkpoint')
    atomic_write(path, lambda stream: torch.save(payload, stream), reserve_bytes = reserve_bytes)
    path = Path(path)
    if path.name != 'last.pt':
        return
    selected = payload['state'].get('best_render') or payload['state'].get('best_map')
    best_path, info_path = path.parent / 'best.pt', path.parent / 'best_info.json'
    if selected is not None:
        info = {key:selected[key] for key in ('stage', 'step', 'total_step', 'metric')}
        previous = json.loads(info_path.read_text(encoding = 'utf-8')) if info_path.exists() else None
        if info != previous or not best_path.exists():
            best = {'schema':'nfplight.best.v1', 'net_g':selected['net_g'], 'selected':info,
                    'clean_profile':payload['clean_profile'], 'identity':payload['identity'],
                    'relation_version':payload['relation_version']}
            atomic_write(best_path, lambda stream: torch.save(best, stream), reserve_bytes = reserve_bytes)
            write_json(info_path, info)
    if backup_dir is None:
        return
    if progress:
        progress('Drive checkpoint backup')
    def copy_to_drive(source, destination):
        def copy(stream):
            with source.open('rb') as original:
                shutil.copyfileobj(original, stream, length = 4 * 1024 * 1024)
        atomic_write(destination, copy)
    try:
        remote_info = backup_dir / 'best_info.json'
        previous = json.loads(remote_info.read_text(encoding = 'utf-8')) if remote_info.exists() else None
        if selected is not None and (info != previous or not (backup_dir / 'best.pt').exists()):
            copy_to_drive(best_path, backup_dir / 'best.pt')
            write_json(remote_info, info)
        copy_to_drive(path, backup_dir / 'last.pt')
        write_json(path.parent / 'backup_status.json', {'ok':True, 'directory':str(backup_dir),
            'total_step':payload['state']['total_step'], 'updated_at':time.time()})
    except (OSError, ValueError) as error:
        write_json(path.parent / 'backup_status.json', {'ok':False, 'directory':str(backup_dir),
            'attempted_total_step':payload['state']['total_step'], 'error':str(error), 'updated_at':time.time()})


def main(argv = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required = True)
    parser.add_argument('--manifest', required = True)
    parser.add_argument('--config', required = True)
    parser.add_argument('--training-config', default = 'configs/training_noquant.json')
    parser.add_argument('--runtime', default = 'configs/runtime_accum4.json')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint-dir')
    args = parser.parse_args(argv)
    root, manifest = map(relative_path, (args.run_root, args.manifest))
    if args.resume:
        relative_path(args.resume)
    for path in (args.config, args.training_config, args.runtime):
        relative_path(path)
    profile = read_config(args.config)
    require(profile['features'].get('synthetic_quantization', 'none') == 'none', 'unquantized experiment profile required')
    require(not root.exists(), 'use a new run root, including for exact resume')
    require(torch.cuda.is_available(), 'select a CUDA GPU runtime; CPU model execution is disabled')
    free, total = torch.cuda.mem_get_info()
    admission = {'gpu_free_bytes': free, 'gpu_total_bytes': total,
                 'host_available_bytes': psutil.virtual_memory().available,
                 'disk_free_bytes': psutil.disk_usage('.').free}
    require(free >= 12 * 2**30 and admission['host_available_bytes'] >= 8 * 2**30
            and admission['disk_free_bytes'] >= 12 * 2**30, f'resource admission failed: {admission}')
    backup_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    if backup_dir is not None:
        require(not backup_dir.is_relative_to(root.resolve()), 'Drive backup must be outside the local run root')
        require(not backup_dir.exists(), 'use a new checkpoint backup folder; existing folders are preserved')
        backup_dir.mkdir(parents = True)
    root.mkdir(parents = True)
    write_json(root / 'admission.json', admission)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text = True).strip()
    write_json(root / 'launch.json', {'commit': revision,
        'dirty_tracked': subprocess.run(['git', 'diff', '--quiet', 'HEAD']).returncode != 0,
        'config': identity(args.config), 'training_config': identity(args.training_config),
        'runtime': identity(args.runtime), 'manifest': identity(manifest), 'argv': vars(args),
        'initialization': 'exact_resume' if args.resume else 'fresh', 'profile': profile, 'sha256_computed': False})
    write_json(root / 'process.json', {'pid': psutil.Process().pid,
        'create_time': psutil.Process().create_time(), 'started_at': datetime.now(timezone.utc).isoformat()})
    activity = {}
    def progress(phase, completed = 0, total = None, **details):
        now = time.time()
        changed = phase != activity.get('phase') or (completed == 0 and activity.get('completed') != 0)
        if changed or not activity:
            activity['started_at'] = now
        if changed or completed in (0, total) or now - activity.get('updated_at', 0) >= 1:
            activity.update(phase = phase, completed = completed, total = total, updated_at = now, **details)
            write_json(root / 'activity.json', activity)
    cli.TrainingPipeline.progress = staticmethod(progress)
    progress('initializing')
    # Same GPU initialization and checkpoint-copy optimization as the active local launcher.
    original_build, original_step = cli.build_model, training.Trainer.step
    def cuda_build(profile):
        progress('model initialization')
        with torch.device('cuda'):
            model = original_build(profile)
        require(all(p.is_cuda and p.dtype == torch.float32 for p in model.parameters()), 'CUDA FP32 model required')
        if profile['profile'] == 'nfplight_32_noquant':
            require(sum(p.numel() for p in model.parameters()) == 64102378,
                    'Colab NFPLight must use the original 64,102,378-parameter model')
        return model
    def timed_step(trainer):
        start = time.perf_counter()
        step = trainer.state['stage_step']
        interval = trainer.policy['evaluation_interval']
        base = step // interval * interval
        cap = trainer.policy[trainer.state['stage'] + '_cap']
        total = min(interval, cap - base) if cap is not None else interval
        progress('training', step - base, total, stage = trainer.state['stage'], stage_step = step)
        row = original_step(trainer)
        row['step_seconds'] = time.perf_counter() - start
        progress('training', row['step'] - base, total, stage = row['stage'], stage_step = row['step'])
        return row
    cli.build_model = cuda_build
    training.Trainer.step = timed_step
    training.save_checkpoint = partial(save_cpu_checkpoint, backup_dir = backup_dir, progress = progress)
    sys.argv = ['nfplight', 'train', '--config', args.config,
        '--training-config', args.training_config, '--runtime', args.runtime,
        '--manifest', str(manifest), '--device', 'cuda', '--out', str(root / 'run')]
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    code = 0
    try:
        cli.main()
    except BaseException:
        code = 1
        raise
    finally:
        write_json(root / 'exit.json', {'exit_code': code, 'finished_at': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
