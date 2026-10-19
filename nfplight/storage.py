from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def write_json(path, value):
    text = json.dumps(value, indent = 2, ensure_ascii = False, allow_nan = False) + '\n'
    atomic_write(path, lambda stream: stream.write(text.encode('utf-8')))


def atomic_write(path, writer, *, reserve_bytes = 0):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    if shutil.disk_usage(path.parent).free < reserve_bytes:
        raise OSError('insufficient disk reserve; previous checkpoint remains unchanged')
    fd, temporary = tempfile.mkstemp(prefix = path.name + '.', suffix = '.tmp', dir = path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, payload, *, reserve_bytes):
    atomic_write(path, lambda stream: torch.save(cpu_tree(payload), stream), reserve_bytes = reserve_bytes)


@contextmanager
def output_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents = True, exist_ok = True)
    lock = directory / '.writer.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode('ascii'))
        os.close(fd)
        yield
    finally:
        lock.unlink()
