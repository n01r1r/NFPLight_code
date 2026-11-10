"""Colab path and report adapters; the model and trainer remain unchanged."""
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
        metrics = ('loss', 'ndrs_sum', 'normal_loss', 'diffuse_loss', 'roughness_loss',
                   'specular_loss', 'render_loss', 'lr', 'gradient_norm', 'step_seconds')
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
        policy = read(self.root / 'run/execution_config.json', {}).get('policy', {})
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
        return {'status':status, 'latest':self.latest, 'train':train, 'validation':self.validation,
                'profile':profile, 'policy':policy, 'fetched_at':datetime.now(timezone.utc).isoformat(),
                'latest_validation':self.validation[-1] if self.validation else None,
                'history_note':'Train: current invocation; validation: checkpoint history included', 'sha256_computed':False}


def plot_training(snapshot):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize = (17, 14), layout = 'constrained')
    train_panels = [(0, ['loss'], 'Train objective'), (1, ['ndrs_sum'], 'Train NDRS L1 sum'),
        (2, ['normal_loss', 'diffuse_loss', 'roughness_loss', 'specular_loss'], 'Train component L1'),
        (3, ['render_loss'], 'Train render L1'), (8, ['lr'], 'Learning rate'),
        (9, ['gradient_norm'], 'Gradient norm before clip'), (10, ['step_seconds'], 'Seconds / optimizer update')]
    for index, metrics, title in train_panels:
        ax = axes.flat[index]
        for stage in ('map', 'render'):
            for metric in metrics:
                points = snapshot['train'].get(stage + ':' + metric, [])
                if not points:
                    continue
                x, raw, smooth = zip(*points)
                line, = ax.plot(x, smooth, label = stage + ':' + metric)
                ax.plot(x, raw, color = line.get_color(), alpha = .18, linewidth = .5)
        ax.set_title(title + ' (raw / mean100)')
        if index == 8 and ax.lines:
            ax.set_yscale('log')
    val_panels = [(4, ['ndrs_sum'], 'Validation NDRS sum', True),
                  (5, ['map_normal', 'map_diffuse', 'map_roughness', 'map_specular'], 'Validation component L1', False),
                  (6, ['render_l1'], 'Validation novel-light render L1', True),
                  (7, ['normal_degrees'], 'Validation normal angle (degrees)', True)]
    for index, metrics, title, by_category in val_panels:
        ax = axes.flat[index]
        for stage in ('map', 'render'):
            for population in (('overall', 'Fabric', 'Leather') if by_category else ('overall',)):
                for metric in metrics:
                    points = []
                    for row in snapshot['validation']:
                        if row['stage'] != stage:
                            continue
                        heldout = row['populations']['heldout']
                        values = heldout['mean'] if population == 'overall' else heldout['category'].get(population, {})
                        if metric in values:
                            points.append((row['total_step'], values[metric]))
                    if points:
                        x, y = zip(*points)
                        ax.plot(x, y, '.-', label = stage + ':' + population + ':' + metric)
        ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel('Total optimizer update')
        ax.grid(alpha = .2)
        if ax.lines:
            ax.legend(fontsize = 7)
    ax = axes.flat[11]
    ax.axis('off')
    state = snapshot['status']
    lines = [f'{key}: {state.get(key)}' for key in ('status', 'stage', 'stage_step', 'total_step', 'process_alive')]
    for stage in ('map', 'render'):
        convergence = state.get('convergence', {}).get(stage, {})
        lines.append(f"{stage}: floor={convergence.get('floor_reached', '—')} bad={convergence.get('floor_bad_observations', '—')}/{snapshot['policy'].get('floor_observations', '—')}")
    lines += ['Train: valid-pixel weighted minibatch', 'Val: crop mean → material macro',
              'Train: current invocation', 'Val: checkpoint history included; missing values omitted']
    ax.text(0, 1, '\n'.join(lines), va = 'top', fontsize = 10)
    fig.suptitle(snapshot['profile'].get('profile', 'Initializing model') + ' · FP32 · optimizer updates', fontsize = 15)
    return fig


def monitor(run_root, process = None, *, refresh_seconds = 10, plots = True):
    if not isinstance(refresh_seconds, (int, float)) or not 0 < refresh_seconds < float('inf'):
        raise ValueError('refresh_seconds must be finite and positive')
    from tqdm.auto import tqdm
    from nfplight.storage import write_json
    reader = TrainingLog(run_root)
    reports = reader.root / 'reports'
    reports.mkdir(exist_ok = True)
    progress, progress_stage, exported, previous = None, None, 0., None
    try:
        while True:
            finished = process is None or process.poll() is not None
            snapshot = reader.snapshot()
            latest = snapshot['latest'] or {}
            stage = latest.get('stage', snapshot['status'].get('stage', 'initial validation'))
            if plots:
                from IPython.display import clear_output, display
                import matplotlib.pyplot as plt
                if progress is not None:
                    progress.close()
                clear_output(wait = True)
                figure = plot_training(snapshot)
                display(figure)
            if plots or stage != progress_stage:
                if progress is not None:
                    progress.close()
                progress = tqdm(total = None, initial = latest.get('step', 0), desc = f'{stage} optimizer updates', unit = 'update')
                progress_stage = stage
            progress.update(max(0, latest.get('step', 0) - progress.n))
            progress.set_postfix(loss = latest.get('loss'), lr = latest.get('lr'), seconds = latest.get('step_seconds'),
                val_step = (snapshot['latest_validation'] or {}).get('total_step'), status = snapshot['status']['status'])
            key = (stage, snapshot['status']['status'], len(snapshot['validation']))
            if finished or key != previous or time.monotonic() - exported >= 60:
                write_json(reports / 'snapshot.json', snapshot)
                if plots:
                    figure.savefig(reports / 'training.png', dpi = 100)
                previous, exported = key, time.monotonic()
            if plots:
                plt.close(figure)
            if finished:
                return snapshot
            time.sleep(refresh_seconds)
    finally:
        if progress is not None:
            progress.close()


def plot_pipeline(profile):
    """Documentation diagram only: no model construction or tensor computation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, axes = plt.subplots(1, 2, figsize = (17, 8), layout = 'constrained')
    ax = axes[0]
    ax.set(xlim = (0, 10), ylim = (0, 10), title = 'NFPLight training pipeline')
    ax.axis('off')
    nested = profile['network'].get('architecture', 'nfplight') == 'nested'
    architecture = 'NAFBlock U-Net++ DS OFF' if nested else 'NFPLight additive U-Net'
    labels = [
        'NPZ material maps: N3 / D3 / R1 / stored S3\n16 crops × 256²; source uint8',
        'MaterialBank: N → unit XYZ, D → linear RGB\nD4 normal transform; S target fixed F0=0.04',
        'canonical_target + valid Nz>0 + roughness floor\nGPU FP32 renderer: near4 / far12 → B×6×256×256',
        'Features: gain9, geometry, relation/log-relation\n9 appearance + 9 log + relation + log-relation + non-sat = 21',
        architecture + '\nraw B×10×256×256 → unit-normal decode',
        'Map: LN + LD + LR + LS\nRender: map sum + 1.0 × rendered linear-HDR L1',
        'valid-pixel weighted micro1 × accum4 → Adam\nheldout material macro → scheduler / best / floor counter',
    ]
    for i, label in enumerate(labels):
        y = 9.3 - 1.35 * i
        ax.add_patch(FancyBboxPatch((.3, y - .45), 9.4, .95, boxstyle = 'round,pad=.08',
                                   facecolor = '#eef5fb', edgecolor = '#3973a0'))
        ax.text(5, y, label, ha = 'center', va = 'center', fontsize = 10)
        if i:
            ax.add_patch(FancyArrowPatch((5, y + .88), (5, y + .53), arrowstyle = '-|>', mutation_scale = 12))
    ax = axes[1]
    if not nested:
        ax.set(xlim = (0, 10), ylim = (0, 10), title = 'Original NFPLight: two encoders / additive skip')
        ax.axis('off')
        dim = profile['network']['dim']
        for x, label in [(2, 'Appearance 18ch'), (7, 'Relation / log / mask 3ch')]:
            for level in range(5):
                ax.text(x, 9 - level * 1.4, f'{label if level == 0 else ""}\nC={dim * 2**level}',
                        ha = 'center', bbox = {'boxstyle':'round,pad=.3', 'fc':'#e6f2ea'})
                if level:
                    ax.annotate('', (x, 9.5 - level * 1.4), (x, 10 - level * 1.4), arrowprops = {'arrowstyle':'->'})
        ax.text(5, 1.4, 'Appearance skips → additive decoder → D3 / S3\nConcat appearance+relation skips → additive decoder → N3 / R1',
                ha = 'center', fontsize = 10)
        fig.suptitle(architecture + ' · FP32 · no observation quantization', fontsize = 15)
        return fig
    ax.set(xlim = (-.7, 4.7), ylim = (-1.9, 5.2), title = 'U-Net++ grid for each decoder (DS OFF)')
    ax.axis('off')
    for col in range(5):
        for level in range(5 - col):
            x, y = col, 4 - level
            ax.text(x, y, f'X{level},{col}', ha = 'center', va = 'center',
                    bbox = {'boxstyle':'round,pad=.35', 'fc':'#e6f2ea' if col == 0 else '#fff0d9'})
            if col == 0 and level < 4:
                ax.add_patch(FancyArrowPatch((x, y - .22), (x, y - .78),
                    arrowstyle = '->', mutation_scale = 12, color = '#777777'))
            if col:
                for earlier in range(col):
                    ax.add_patch(FancyArrowPatch((earlier + .20, y - .12), (x - .22, y - .12),
                        arrowstyle = '->', mutation_scale = 9, color = '#b58b4b', alpha = .5))
                ax.add_patch(FancyArrowPatch((col - 1 + .15, y - 1 + .22), (x - .12, y - .18),
                    arrowstyle = '->', mutation_scale = 12, color = '#3973a0'))
    channels = [profile['network']['dim'] * 2**level for level in range(5)]
    ax.text(0, -.75, f'Xl,0 = encoder / middle\nAppearance: C={channels}\nRelation: same C; N/R decoder input concat → 2C', fontsize = 10)
    ax.text(0, -1.6, 'Appearance decoder → D3 / S3; concatenated decoder → N3 / R1\nX0,4 only supervised. Fusion: 1×1 projection + 2 NAFBlocks.', fontsize = 10)
    fig.suptitle(f"NAFBlock U-Net++ · dim{profile['network']['dim']} · FP32 · no observation quantization", fontsize = 15)
    return fig
