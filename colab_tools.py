"""Drive paths, process launch, and the two loss plots used by Colab."""
from collections import Counter
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import time


def relative_path(value):
    path = Path(value)
    if (path.is_absolute() or PureWindowsPath(str(value)).root or PureWindowsPath(str(value)).drive
            or PurePosixPath(str(value).replace('\\', '/')).is_absolute()
            or '..' in PurePosixPath(str(value).replace('\\', '/')).parts):
        raise ValueError('use a project-relative path without ..')
    return path


def write_once(path, text):
    path = relative_path(path)
    data = text.encode('utf-8')
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'refusing to replace different content: {path}')
        return path
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open('xb') as stream:
        stream.write(data)
    return path


def attach_drive_dataset(source, destination):
    source, destination = Path(source).resolve(), relative_path(destination)
    if not source.is_dir() or not (source / 'source_shards').is_dir():
        raise FileNotFoundError(f'expected Drive dataset folder with source_shards: {source}')
    if not (source / 'manifests/fabric_leather_training_manifest.jsonl').is_file():
        raise FileNotFoundError('Drive dataset manifest missing')
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(f'refusing to replace dataset path: {destination}')
        return destination
    destination.parent.mkdir(parents = True, exist_ok = True)
    destination.symlink_to(source, target_is_directory = True)
    return destination


def relocate_manifest(manifest, data_root, output, *, expected_counts = None):
    """Keep row order/identities, exclude test, resolve shard paths relative to cwd."""
    manifest, data_root, output = map(relative_path, (manifest, data_root, output))
    rows = [json.loads(line) for line in manifest.read_text(encoding = 'utf-8').splitlines() if line.strip()]
    selected, counts, ids, groups = [], Counter(), set(), {'train': set(), 'val': set()}
    sizes, paths = Counter(), set()
    for row in rows:
        if row['split'] == 'test':
            continue
        if row['split'] not in groups or row['category'] not in ('Fabric', 'Leather'):
            raise ValueError('unexpected training split/category')
        if row['material_id'] in ids:
            raise ValueError('duplicate material id')
        ids.add(row['material_id'])
        groups[row['split']].add(row['group_id'])
        parts = PurePosixPath(row['shard_path'].replace('\\', '/')).parts
        if parts.count('source_shards') != 1 or '..' in parts:
            raise ValueError('expected one source_shards anchor without path traversal')
        path = data_root.joinpath(*parts[parts.index('source_shards'):])
        if not path.resolve().is_relative_to(data_root.resolve()) or path.suffix != '.npz':
            raise ValueError('shard escapes dataset root or is not NPZ')
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in paths:
            raise ValueError('duplicate shard path')
        paths.add(path)
        # Unused Windows source_dir and historical digest metadata are not execution inputs.
        portable = {key: row[key] for key in ('group_id', 'material_id', 'category', 'split')}
        portable['shard_path'] = path.as_posix()
        selected.append(portable)
        key = row['split'] + '/' + row['category']
        counts[key] += 1
        sizes[key] += path.stat().st_size
    if groups['train'] & groups['val']:
        raise ValueError('train/validation source group overlap')
    if not groups['train'] or not groups['val']:
        raise ValueError('empty train/validation population')
    if expected_counts is not None and dict(counts) != expected_counts:
        raise ValueError(f'population mismatch: {dict(counts)} != {expected_counts}')
    text = ''.join(json.dumps(row, ensure_ascii = False) + '\n' for row in selected)
    write_once(output, text)
    return {'manifest': output.as_posix(), 'path_base': 'project working directory, not manifest parent',
            'materials': dict(counts), 'bytes': dict(sizes), 'total_bytes': sum(sizes.values()),
            'official_test_payloads_read': 0, 'sha256_computed': False}


class TrainingLog:
    def __init__(self, run_root):
        from collections import deque
        self.root = relative_path(run_root)
        if not (self.root / 'process.json').is_file():
            raise FileNotFoundError('training process metadata not created')
        self.offset, self.anchor, self.file_id = 0, b'', None
        self.series, self.windows, self.latest = {}, {}, None
        self.validation, self.validation_stamp = [], None
        self.deque = deque

    def snapshot(self):
        import math
        from datetime import datetime, timezone
        import psutil
        def read(path, default):
            return json.loads(path.read_text(encoding = 'utf-8')) if path.exists() else default
        path = self.root / 'run/train.jsonl'
        metrics = ('loss',)
        if path.exists():
            stat = path.stat()
            file_id = (stat.st_dev, stat.st_ino)
            with path.open('rb') as stream:
                stream.seek(max(0, self.offset - len(self.anchor)))
                changed = self.anchor and stream.read(len(self.anchor)) != self.anchor
                if self.file_id != file_id or stat.st_size < self.offset or changed:
                    self.offset, self.anchor = 0, b''
                    self.series, self.windows, self.latest = {}, {}, None
                self.file_id = file_id
                stream.seek(self.offset)
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b'\n'):
                        stream.seek(start)
                        break
                    row = json.loads(line)
                    if (row.get('stage') not in ('map', 'render') or type(row.get('total_step')) is not int
                            or type(row.get('step')) is not int or row['total_step'] < 0 or row['step'] < 0):
                        raise ValueError('invalid training log stage/update')
                    for metric in metrics:
                        value = row.get(metric)
                        if value is None:
                            continue
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                            raise ValueError(f'invalid training metric: {metric}')
                        key = row['stage'] + ':' + metric
                        window = self.windows.setdefault(key, self.deque(maxlen = 100))
                        window.append(value)
                        self.series.setdefault(key, []).append([row['total_step'], value, sum(window) / len(window)])
                    self.latest = row
                    self.offset = stream.tell()
                stream.seek(max(0, self.offset - 128))
                self.anchor = stream.read(min(128, self.offset))
        path = self.root / 'run/evaluation.json'
        stamp = (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
        if stamp != self.validation_stamp:
            self.validation = read(path, [])
            from nfplight.training import finite_metrics
            finite_metrics(self.validation)
            self.validation_stamp = stamp
        status = read(self.root / 'run/status.json', {'status':'initial_validation'})
        process = read(self.root / 'process.json', {})
        exited = read(self.root / 'exit.json', None)
        alive = False
        try:
            proc = psutil.Process(process['pid'])
            alive = proc.is_running() and abs(proc.create_time() - process['create_time']) < .01
        except (psutil.Error, KeyError):
            pass
        status = dict(status, process_alive = alive, pid = process.get('pid'))
        if self.latest:
            status.update(total_step = self.latest['total_step'], stage = self.latest['stage'], stage_step = self.latest['step'])
        if exited and exited['exit_code']:
            status['status'] = 'failed'
        elif not alive and status['status'] not in ('complete', 'budget_paused', 'memory_paused'):
            status['status'] = 'process_missing_cause_unknown'
        implementation = read(self.root / 'run/implementation.json', {})
        launch = read(self.root / 'launch.json', {})
        profile = implementation.get('profile', launch.get('profile', {}))
        execution = read(self.root / 'run/execution_config.json', {})
        policy = execution.get('policy', {})
        train = {}
        for key, points in self.series.items():
            if len(points) <= 6000:
                train[key] = points
                continue
            chosen = set()
            width = math.ceil(len(points) / 1500)
            for start in range(0, len(points), width):
                block = points[start:start + width]
                chosen.update((start, start + len(block) - 1))
                for column in (1, 2):
                    chosen.add(start + min(range(len(block)), key = lambda i: block[i][column]))
                    chosen.add(start + max(range(len(block)), key = lambda i: block[i][column]))
            train[key] = [points[index] for index in sorted(chosen)]
        return {'activity':read(self.root / 'activity.json', {}),
                'backup':read(self.root / 'run/backup_status.json', None),
                'process_started_at':process.get('create_time', time.time()),
                'render_weight':execution.get('scientific_settings', {}).get('loss', {}).get('render_weight', 1.),
                'status':status, 'latest':self.latest, 'train':train, 'validation':self.validation,
                'profile':profile, 'policy':policy, 'fetched_at':datetime.now(timezone.utc).isoformat(),
                'latest_validation':self.validation[-1] if self.validation else None,
                'history_note':'Train: current invocation; validation: checkpoint history included', 'sha256_computed':False}


def validation_loss(row, weight = 1.):
    mean = row['populations']['heldout']['mean']
    return mean['ndrs_sum'] + (weight * mean['render_l1'] if row['stage'] == 'render' else 0.)


def plot_training(snapshot):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize = (11, 3.5), layout = 'constrained')
    for stage in ('map', 'render'):
        points = snapshot['train'].get(stage + ':loss', [])
        if points:
            x, raw, mean = zip(*points)
            line, = axes[0].plot(x, mean, '.-', markersize = 3, label = stage)
            axes[0].plot(x, raw, color = line.get_color(), alpha = .2, linewidth = .5)
        rows = [row for row in snapshot['validation'] if row['stage'] == stage]
        if rows:
            axes[1].plot([row['total_step'] for row in rows],
                [validation_loss(row, snapshot['render_weight']) for row in rows], 'o-', markersize = 4, label = stage)
    for ax, title, waiting in zip(axes, ('Train loss', 'Validation loss'),
                                ('Waiting for first optimizer update', 'Waiting for completed validation')):
        ax.set(title = title, xlabel = 'Optimizer update', ylabel = 'Loss (log scale)')
        ax.set_yscale('log', nonpositive = 'mask')
        ax.grid(alpha = .2)
        if ax.lines:
            ax.legend()
            if any(value == 0 for line in ax.lines for value in line.get_ydata()):
                ax.text(.02, .02, 'Zero loss cannot appear on log axis', transform = ax.transAxes, fontsize = 8)
        else:
            ax.text(.5, .5, waiting, ha = 'center', transform = ax.transAxes)
    return fig


def launch_training(config, manifest, run_root, resume = None, checkpoint_dir = None):
    import subprocess
    import psutil
    config, manifest, run_root = map(relative_path, (config, manifest, run_root))
    if run_root.exists():
        raise FileExistsError(f'Choose a new run name: {run_root}')
    for metadata in Path('runs').glob('*/process.json'):
        record = json.loads(metadata.read_text(encoding = 'utf-8'))
        try:
            child = psutil.Process(record['pid'])
            if child.is_running() and abs(child.create_time() - record['create_time']) < .01:
                raise RuntimeError(f'Trainer PID {child.pid} is running. Use the Live view cell.')
        except psutil.NoSuchProcess:
            pass
    if resume is not None:
        relative_path(resume)
    Path('work').mkdir(exist_ok = True)
    command = [sys.executable, '-u', 'train_colab.py', '--config', str(config),
               '--manifest', str(manifest), '--run-root', str(run_root)]
    if resume is not None:
        command.extend(['--resume', str(resume)])
    if checkpoint_dir is not None:
        command.extend(['--checkpoint-dir', str(checkpoint_dir)])
    stdout, stderr = Path('work') / (run_root.name + '.stdout.log'), Path('work') / (run_root.name + '.stderr.log')
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.Popen(command, stdout = out, stderr = err)
    print(f'Started PID {process.pid}. Loading PyTorch and checking resources...', flush = True)
    try:
        while process.poll() is None and not (run_root / 'process.json').exists():
            time.sleep(.2)
        if process.poll() is not None:
            raise RuntimeError(stderr.read_text(encoding = 'utf-8', errors = 'replace')[-8000:])
    except BaseException:
        print(f'Child PID {process.pid}; still running: {process.poll() is None}', flush = True)
        raise
    return process


def monitor(run_root, process = None, *, refresh_seconds = 5, plots = True, follow = True):
    if not isinstance(refresh_seconds, (int, float)) or not 0 < refresh_seconds < float('inf'):
        raise ValueError('refresh_seconds must be finite and positive')
    from tqdm import tqdm
    from IPython.display import clear_output, display
    reader = TrainingLog(run_root)
    try:
        while True:
            finished = process is not None and process.poll() is not None
            snapshot = reader.snapshot()
            finished = finished or not snapshot['status']['process_alive']
            activity = snapshot['activity']
            latest = snapshot['latest'] or {}
            phase = activity.get('phase')
            done, total = activity.get('completed', 0), activity.get('total')
            if phase is None:
                if latest:
                    phase = 'training / validation (legacy run)'
                else:
                    phase = 'model initialization / initial validation (legacy run; no crop progress)'
            clear_output(wait = True)
            print(f"{phase} | {snapshot['status']['status']} | PID {snapshot['status'].get('pid')} | alive={snapshot['status']['process_alive']} | "
                  f"last logged update={latest.get('total_step', 'none')} | completed validations={len(snapshot['validation'])}", flush = True)
            elapsed = max(0., time.time() - snapshot['process_started_at'])
            stage = activity.get('stage', latest.get('stage', snapshot['status'].get('stage', 'map')))
            step = activity.get('stage_step', latest.get('step', 0))
            if latest.get('stage') == stage:
                step = max(step, latest['step'])
            cap = snapshot['policy'].get(stage + '_cap')
            if cap:
                print(tqdm.format_meter(step, cap, elapsed, prefix = stage + ' train step (cap)',
                    bar_format = '{desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]'))
            else:
                print(f'{stage} train step: {step} | elapsed {elapsed:.0f}s')
            if phase == 'validation':
                print(f'Val crops: {done}/{total}. Final loss after the full pass.')
            if latest:
                print(f"Train loss: {latest['loss']:.6g} | lr: {latest.get('lr', 'unrecorded')}")
            if activity.get('updated_at'):
                print(f"Last progress: {max(0., time.time() - activity['updated_at']):.0f}s ago")
            backup = snapshot['backup']
            if backup:
                print(f"Drive backup step {backup['total_step']} | {backup['directory']}" if backup['ok']
                      else f"Drive backup failed: {backup['error']} | local checkpoint preserved")
            if plots:
                import matplotlib.pyplot as plt
                figure = plot_training(snapshot)
                display(figure)
                plt.close(figure)
            if finished:
                if process is not None and process.returncode:
                    error_path = Path('work') / (reader.root.name + '.stderr.log')
                    raise RuntimeError(error_path.read_text(encoding = 'utf-8', errors = 'replace')[-8000:])
                return snapshot
            if not follow:
                return snapshot
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        print('Viewer interrupted. Trainer is unchanged; use the Live cell to reconnect.', flush = True)
        raise
