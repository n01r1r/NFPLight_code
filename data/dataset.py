import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data as data

from utils import svBRDF, FileClient, imfrombytes


class SynDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = {'type': 'disk'}
        svbrdf_folder = opt['svbrdf_root']
        svbrdf_names = sorted(os.listdir(svbrdf_folder))
        self.svbrdf_paths = [os.path.join(svbrdf_folder, name) for name in svbrdf_names]
        self.svBRDF_utils = svBRDF()
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

    def __getitem__(self, index):
        img_bytes = self.file_client.get(self.svbrdf_paths[index], 'brdf')
        img = imfrombytes(img_bytes, float32=True)[:, :, ::-1]
        return {
            'svbrdfs': self.svBRDF_utils.get_svbrdfs(img),
            'name': os.path.basename(self.svbrdf_paths[index]),
        }

    def __len__(self):
        return len(self.svbrdf_paths)


class RealDataset(data.Dataset):
    """Load paired near/far captures into float32 NCHW linear RGB tensors.

    ``linear_rgb`` accepts float32 HWC ``near.npy``/``far.npy`` or uint16 PNG
    pairs and performs no transfer-function conversion.  ``gamma22`` is the
    legacy encoded-PNG mode; it decodes the repository's power-2.2 transfer
    before resizing.  The old ``input_gamma=True`` option selects gamma22.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.input_mode = _resolve_input_mode(opt)
        root = Path(opt['root'])
        if not root.is_dir():
            raise FileNotFoundError(f'capture root does not exist: {root}')
        self.svbrdf_paths = sorted(path for path in root.iterdir() if path.is_dir())
        if not self.svbrdf_paths:
            raise ValueError(f'capture root contains no capture directories: {root}')

    def __getitem__(self, index):
        capture = self.svbrdf_paths[index]
        near_path = _capture_file(capture, 'near', self.input_mode)
        far_path = _capture_file(capture, 'far', self.input_mode)
        if near_path.suffix != far_path.suffix:
            raise ValueError(
                f'near/far capture formats must match: {near_path.name}, {far_path.name}'
            )
        near = _read_capture(near_path, self.input_mode)
        far = _read_capture(far_path, self.input_mode)
        _check_pair(near, far, near_path, far_path)

        size = self.opt.get('imageSize', self.opt.get('image_size'))
        if size not in (None, False):
            if self.input_mode == 'linear_rgb' and near.shape[0] != near.shape[1]:
                raise ValueError('RAW inference requires a rectified square patch; resize must not change aspect ratio')
            size = int(size)
            if size <= 0:
                raise ValueError(f'imageSize must be positive, got {size}')
            near = _resize_linear(near, size)
            far = _resize_linear(far, size)
            _check_linear(near, near_path)
            _check_linear(far, far_path)

        metadata, metadata_path = _metadata(capture, self.input_mode)
        hashes = {'near': _sha256(near_path), 'far': _sha256(far_path)}
        if metadata_path is not None:
            hashes['metadata'] = _sha256(metadata_path)
        inputs = torch.from_numpy(np.concatenate((near, far), axis=2))
        inputs = inputs.permute(2, 0, 1).contiguous().to(dtype=torch.float32)
        return {
            'inputs': inputs,
            'name': capture.name,
            'metadata': metadata,
            'input_hashes': hashes,
        }

    def __len__(self):
        return len(self.svbrdf_paths)


def _resolve_input_mode(opt):
    mode = opt.get('input_mode')
    if mode is None:
        mode = 'gamma22' if opt.get('input_gamma', False) else 'linear_rgb'
    if mode not in ('linear_rgb', 'gamma22'):
        raise ValueError(
            f"unsupported RealDataset input_mode {mode!r}; expected 'linear_rgb' or 'gamma22'"
        )
    return mode


def _capture_file(capture, stem, mode):
    npy = capture / f'{stem}.npy'
    png = capture / f'{stem}.png'
    if npy.is_file() and png.is_file():
        raise ValueError(f'ambiguous {stem} capture in {capture}: both .npy and .png exist')
    if mode == 'gamma22':
        if png.is_file():
            return png
        raise FileNotFoundError(f'missing {stem}.png capture in {capture}')
    if npy.is_file():
        return npy
    if png.is_file():
        return png
    raise FileNotFoundError(f'missing {stem}.npy or {stem}.png capture in {capture}')


def _read_capture(file_path, mode):
    if file_path.suffix == '.npy':
        try:
            array = np.load(file_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f'could not read NumPy capture {file_path}: {error}') from error
        if array.dtype != np.float32:
            raise TypeError(f'{file_path} must be float32 HWC RGB; observed {array.dtype}')
        _check_linear(array, file_path)
        return np.ascontiguousarray(array)

    raw = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f'could not decode PNG capture: {file_path}')
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(f'{file_path} must be a 3-channel HWC RGB PNG; observed {raw.shape}')
    if raw.dtype not in (np.uint8, np.uint16):
        raise TypeError(f'{file_path} must be uint8 or uint16 PNG; observed {raw.dtype}')
    if mode == 'linear_rgb' and raw.dtype != np.uint16:
        raise TypeError(f'linear_rgb PNG must be uint16: {file_path}')
    divisor = 65535.0 if raw.dtype == np.uint16 else 255.0
    linear = raw[:, :, ::-1].astype(np.float32) / divisor
    if mode == 'gamma22':
        linear = np.power(linear, np.float32(2.2)).astype(np.float32)
    _check_linear(linear, file_path)
    return np.ascontiguousarray(linear)


def _check_linear(array, file_path):
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f'{file_path} must have HWC RGB shape [H,W,3]; observed {array.shape}')
    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f'{file_path} has invalid image size {(height, width)}')
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f'{file_path} decoded array must be floating point')
    if not np.isfinite(array).all():
        raise ValueError(f'{file_path} contains non-finite linear RGB values')
    low, high = float(array.min()), float(array.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(
            f'{file_path} linear RGB must be in [0,1]; observed [{low:.6g},{high:.6g}]'
        )


def _check_pair(near, far, near_path, far_path):
    if near.shape[0] != far.shape[0]:
        raise ValueError(
            f'near/far height mismatch: {near_path.name}={near.shape[0]}, '
            f'{far_path.name}={far.shape[0]}'
        )
    if near.shape[1] != far.shape[1]:
        raise ValueError(
            f'near/far width mismatch: {near_path.name}={near.shape[1]}, '
            f'{far_path.name}={far.shape[1]}'
        )


def _resize_linear(array, size):
    if array.shape[:2] == (size, size):
        return np.ascontiguousarray(array, dtype=np.float32)
    return np.ascontiguousarray(cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA), dtype=np.float32)


def _sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _metadata(capture, mode):
    path = capture / 'metadata.json'
    if not path.is_file():
        if mode == 'linear_rgb':
            raise FileNotFoundError(f'RAW linear_rgb capture {capture.name} requires metadata.json')
        return {}, None
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid metadata JSON {path}: {error}') from error
    if not isinstance(document, dict):
        raise ValueError(f'capture metadata must be a JSON object: {path}')
    if mode == 'linear_rgb':
        _check_raw_metadata(document, path)
    return document, path


def _check_raw_metadata(metadata, path):
    if metadata.get('working_space') != 'linear_srgb':
        raise ValueError(f'{path} must declare working_space=linear_srgb')
    try:
        near = float(metadata['near_distance'])
        far = float(metadata['far_distance'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'{path} must declare near_distance and far_distance') from error
    if not (math.isclose(near, 2.414, rel_tol=1e-6, abs_tol=1e-6) and
            math.isclose(far, 10.0, rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError(f'{path} must declare calibrated near=2.414 and far=10')
    scale = metadata.get('reference_scale')
    try:
        # Inputs are already mapped to the training reference.  The model applies
        # its fixed geometry ratio itself, so a metadata scale must be exactly 1.
        scale_ok = scale is not None and math.isclose(
            float(scale), 1.0, rel_tol=1e-6, abs_tol=1e-6
        )
    except (TypeError, ValueError):
        scale_ok = False
    equal_reference = (
        metadata.get('reference_exposure') == 'equal' and
        metadata.get('reference_flux') == 'equal'
    )
    if scale is not None and not scale_ok:
        raise ValueError(f'{path} reference_scale must be exactly 1')
    if not equal_reference:
        raise ValueError(
            f'{path} must declare equal reference_exposure/reference_flux'
        )
    if metadata.get('preprocessing_done_externally') is not True:
        raise ValueError(f'{path} must declare preprocessing_done_externally=true')


def log_normalization(img, eps=1e-2):
    return (torch.log(img+eps)-torch.log(torch.ones((1,))*eps))/(torch.log(1+torch.ones((1,))*eps)-torch.log(torch.ones((1,))*eps))
