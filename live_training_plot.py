"""Read-only training plots: python live_training_plot.py --run-dir <run>.

Display contract: docs/TRAINING_HTML_REPORT_CONTRACT.md.
Reads existing logs only. Does not train, score test data, or hash files.
"""
import argparse
from datetime import datetime, timezone
from collections import deque
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def plateau(history, stage):
    rows = [row for row in history if row.get('stage') == stage]
    values = []
    for row in rows:
        if stage == 'map':
            parts = [row.get('map_' + name) for name in ('normal', 'diffuse', 'roughness')]
            value = sum(parts) / 3 if all(numeric(x) for x in parts) else row.get('ndr_mean')
        else:
            value = row.get('render_l1')
        if not numeric(value):
            return {'status': 'missing_or_invalid_metric', 'streak': 0}
        values.append(value)
    improvements = [(a-b) / max(abs(a), 1e-12) for a,b in zip(values, values[1:])]
    streak = 0
    for improvement in reversed(improvements):
        if improvement >= .005:
            break
        streak += 1
    return {'status': 'plateau' if streak >= 10 else 'observing', 'streak': streak,
            'relative_improvement': improvements[-1] if improvements else None,
            'automatic_stop': False}


def display_points(points, column, limit=6000):
    """Reduce once from complete history, independently for each plotted value."""
    if len(points) <= limit:
        return points
    width = math.ceil(len(points) / (limit // 4))
    result = []
    for start in range(0, len(points), width):
        block = points[start:start + width]
        chosen = {0, len(block) - 1,
                  min(range(len(block)), key=lambda i: block[i][column]),
                  max(range(len(block)), key=lambda i: block[i][column])}
        result.extend(block[i] for i in sorted(chosen))
    return result


class LogReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.offset = 0
        self.identity = None
        self.anchor = b''
        self.series = {}
        self.windows = {}
        self.count = 0
        self.latest = None
        self.errors = 0

    def reset(self):
        self.offset = 0
        self.anchor = b''
        self.series = {}
        self.windows = {}
        self.count = 0
        self.latest = None
        self.errors = 0

    def update(self):
        path = self.directory / 'train.jsonl'
        if not path.exists():
            return
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        with path.open('rb') as stream:
            stream.seek(max(0, self.offset-len(self.anchor)))
            changed = self.anchor and stream.read(len(self.anchor)) != self.anchor
            if identity != self.identity or stat.st_size < self.offset or changed:
                self.reset()
            self.identity = identity
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    stream.seek(start)
                    break
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("train row must be an object")
                    step = row.get('total_step')
                    if not numeric(step):
                        raise ValueError('missing finite total_step')
                    stage = row.get('stage', 'unknown')
                    for key in ('loss', 'map_loss', 'ndrs_sum', 'ndr_mean', 'normal_loss', 'diffuse_loss', 'roughness_loss', 'specular_loss', 'render_loss', 'lr'):
                        value = row.get(key)
                        if value is None:
                            continue
                        if not numeric(value):
                            raise ValueError('non-finite ' + key)
                    for key in ('loss', 'map_loss', 'ndrs_sum', 'ndr_mean', 'normal_loss', 'diffuse_loss', 'roughness_loss', 'specular_loss', 'render_loss', 'lr'):
                        value = row.get(key)
                        if not numeric(value):
                            continue
                        name = stage + ':' + key
                        window = self.windows.setdefault(name, deque(maxlen=100))
                        window.append(value)
                        points = self.series.setdefault(name, [])
                        points.append([step, value, sum(window)/len(window)])
                    self.count += 1
                    self.latest = row
                except (ValueError, TypeError, UnicodeDecodeError):
                    self.errors += 1
                self.offset = stream.tell()
            stream.seek(max(0,self.offset-128))
            self.anchor = stream.read(min(128,self.offset))

    def snapshot(self):
        self.update()
        validation = self.directory / 'validation.json'
        error = None
        try:
            document = json.loads(validation.read_text(encoding='utf-8')) if validation.exists() else {}
            if not isinstance(document, dict) or not isinstance(document.get('history', []), list):
                raise ValueError('validation must contain a history array')
            for row in document.get('history', []):
                if not isinstance(row, dict) or not isinstance(row.get('category', {}), dict) or not all(isinstance(x, dict) for x in row.get('category', {}).values()):
                    raise ValueError('validation rows and category metrics must be objects')
        except (ValueError, OSError) as exc:
            document = {}
            error = str(exc)
        history = document.get('history', [])
        valseries = {}
        for row in history:
            stage = row.get('stage', 'unknown')
            step = row.get('total_step', row.get('step'))
            if not numeric(step):
                continue
            # Prefer recorded global updates; never guess missing global offsets.
            for key in ('map_normal', 'map_diffuse', 'map_roughness', 'map_specular', 'ndrs_sum', 'render_l1', 'normal_degrees'):
                value = row.get(key)
                if numeric(value):
                    valseries.setdefault(stage+':'+key, []).append([step,value])
            for category, metrics in row.get('category', {}).items():
                for key in ('map_normal','map_diffuse','map_roughness','map_specular','ndr_mean','ndrs_sum','render_l1','normal_degrees','normal_degrees_pixel_median','normal_degrees_pixel_p95'):
                    if numeric(metrics.get(key)):
                        valseries.setdefault(stage+':'+category+':'+key, []).append([step,metrics[key]])
            parts = [row.get('map_'+key) for key in ('normal','diffuse','roughness')]
            if all(numeric(x) for x in parts):
                valseries.setdefault(stage+':ndr_mean', []).append([step,sum(parts)/3])
        try:
            status = json.loads((self.directory/'status.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            status = {}
        timestamps = {}
        for name, filename in [('train_log', 'train.jsonl'), ('validation_file', 'validation.json'), ('status_file', 'status.json')]:
            path = self.directory / filename
            timestamps[name] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None
        return {'timestamps': timestamps, 'latest_validation': history[-1] if history else None, 'run_dir': str(self.directory), 'train': {key: display_points(points, 1) for key, points in self.series.items()},
                'train_smooth': {key: display_points(points, 2) for key, points in self.series.items()}, 'val': valseries,
                'status': status, 'val_axis': 'total optimizer update' if all('total_step' in row for row in history) else 'stage-local optimizer update',
                'rows_read': self.count, 'invalid_train_rows': self.errors,
                'latest': self.latest, 'validation_error': error,
                'plateau': {stage:plateau(history,stage) for stage in ('map','render')},
                'note': 'Train uses total_step; val uses recorded total_step or legacy stage-local step. Raw and 100-update mean are independently display-decimated from complete history; logs remain complete. Missing component metrics are not inferred.'}


HTML_PATH = Path(__file__).with_suffix('.html')


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, reader, **kwargs):
        self.reader = reader
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            payload, content_type = HTML_PATH.read_bytes(), 'text/html; charset=utf-8'
        elif self.path == '/api':
            try:
                payload = json.dumps(self.reader.snapshot(), allow_nan=False).encode()
            except (ValueError, OSError, TypeError) as exc:
                self.send_error(503,str(exc))
                return
            content_type = 'application/json'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type',content_type)
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self,*args):
        pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    # Single-threaded reads keep the incremental log cache coherent.
    server=HTTPServer(('127.0.0.1',args.port),partial(Handler,reader=LogReader(args.run_dir.resolve())))
    print(f'Live plot: http://127.0.0.1:{args.port}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':
    main()
