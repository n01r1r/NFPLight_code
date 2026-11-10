"""Explicit fresh/strict-resume Colab entry point for the unchanged frozen CLI."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
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


def save_cpu_checkpoint(path, payload, *, reserve_bytes):
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
    atomic_write(path, lambda stream: torch.save(payload, stream), reserve_bytes = reserve_bytes)


def main(argv = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required = True)
    parser.add_argument('--manifest', required = True)
    parser.add_argument('--config', required = True)
    parser.add_argument('--training-config', default = 'configs/training_noquant.json')
    parser.add_argument('--runtime', default = 'configs/runtime_accum4.json')
    parser.add_argument('--resume')
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
    # Same GPU initialization and checkpoint-copy optimization as the active local launcher.
    original_build, original_step = cli.build_model, training.Trainer.step
    def cuda_build(profile):
        with torch.device('cuda'):
            model = original_build(profile)
        require(all(p.is_cuda and p.dtype == torch.float32 for p in model.parameters()), 'CUDA FP32 model required')
        return model
    def timed_step(trainer):
        start = time.perf_counter()
        row = original_step(trainer)
        row['step_seconds'] = time.perf_counter() - start
        return row
    cli.build_model = cuda_build
    training.Trainer.step = timed_step
    training.save_checkpoint = save_cpu_checkpoint
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
