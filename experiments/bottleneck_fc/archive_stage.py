"""Preserve map-stage boundary checkpoints while the pilot trainer runs."""
from pathlib import Path
import argparse
import io
import json
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from experiments.bottleneck_fc.config import atomic_write_json, replace_with_retry, process_identity_matches, resolve_repo_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = resolve_repo_path(args.run_dir)
    execution = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    settings = execution['settings']
    boundary = settings['training']['map_updates']
    seeds = settings['training']['seeds']
    expected = [root / f'seed{seed}_baseline' for seed in seeds]
    result = {'status': 'watching', 'map_boundary': boundary, 'saved': []}
    try:
        while True:
            for seed, folder in zip(seeds, expected):
                target = folder / 'map_stage_end.pt'
                if target.exists():
                    continue
                status_path = folder / 'status.json'
                if not status_path.exists():
                    continue
                status = json.loads(status_path.read_text(encoding='utf-8'))
                if status.get('last_committed_step') != boundary:
                    continue
                checkpoint = folder / 'last.pt'
                payload = checkpoint.read_bytes()
                state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
                if (state['step'], state['seed'], state['arm_id']) != (boundary, seed, 'baseline'):
                    continue
                identity = state['execution_identity']
                for key in ('settings', 'file_identities', 'config_source'):
                    if identity[key] != execution[key]:
                        raise ValueError(f'checkpoint execution mismatch: {key}')
                temporary = target.with_suffix('.pt.archive.tmp')
                temporary.write_bytes(payload)
                replace_with_retry(temporary, target)
                result['saved'].append(str(target))
                print(f'archived seed={seed} map_step={boundary}: {target}', flush=True)
                del state, payload
            if all((folder / 'map_stage_end.pt').is_file() for folder in expected):
                result['status'] = 'complete'
                atomic_write_json(root / 'stage_archive_status.json', result)
                return
            atomic_write_json(root / 'stage_archive_status.json', result)
            current = json.loads((root / 'status.json').read_text(encoding='utf-8'))
            if current.get('status') in ('complete', 'failed'):
                raise RuntimeError('pilot stopped before all map-stage checkpoints were archived')
            if current.get('status') == 'running' and not process_identity_matches(current.get('pid'), current.get('process_identity')):
                raise RuntimeError('pilot process exited before stage archives completed')
            time.sleep(2)
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(root / 'stage_archive_status.json', result)
        raise


if __name__ == '__main__':
    main()
