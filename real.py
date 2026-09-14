import argparse
import hashlib
import json
import torch
import os
from os import path as osp
from data.dataset import RealDataset
from model.nfplight_matsynth_capture_model import (
    EXPECTED_INPUT_SHAPE,
    EXPECTED_OUTPUT_SHAPE,
    INPUT_RANGE_TOLERANCE,
    MatSynthCaptureModel,
    normalize_checkpoint_state_dict,
)
from model.nfplight_model import checkpoint_feature_version


CHECKPOINT_INPUT_CHANNELS = {
    "matsynth21": 18,
    "legacy33": 30,
}


def detect_estimator_family(loadpath_network_g):
    """Infer estimator family from ``intro_albedo.weight`` input channels.

    The synthetic 21-channel estimator has 18 albedo-side channels and the
    legacy real 33-channel estimator has 30.  Any other checkpoint is rejected
    before allocating a model so a mismatched architecture cannot be hidden by
    a non-strict load.
    """
    if not loadpath_network_g:
        raise ValueError("--loadpath_network_g is required and must name a checkpoint")
    if not os.path.isfile(loadpath_network_g):
        raise FileNotFoundError(
            f"network checkpoint does not exist: {loadpath_network_g}"
        )
    checkpoint = torch.load(loadpath_network_g, map_location="cpu", weights_only=False)
    state_dict = normalize_checkpoint_state_dict(checkpoint)
    candidates = [
        value for key, value in state_dict.items()
        if key.endswith("intro_albedo.weight") and torch.is_tensor(value)
    ]
    if len(candidates) != 1 or candidates[0].ndim != 4:
        raise ValueError(
            "network checkpoint must contain exactly one 4D intro_albedo.weight"
        )
    input_channels = int(candidates[0].shape[1])
    for family, expected_channels in CHECKPOINT_INPUT_CHANNELS.items():
        if input_channels == expected_channels:
            return family
    supported = ", ".join(
        f"{family}={channels} channels"
        for family, channels in CHECKPOINT_INPUT_CHANNELS.items()
    )
    raise ValueError(
        f"unsupported intro_albedo input channels: {input_channels}; expected {supported}"
    )


def resolve_estimator_family(requested_family, loadpath_network_g):
    """Resolve and validate the requested family against checkpoint structure."""
    detected_family = detect_estimator_family(loadpath_network_g)
    if requested_family == "auto":
        return detected_family
    if requested_family not in CHECKPOINT_INPUT_CHANNELS:
        raise ValueError(f"unsupported estimator family: {requested_family}")
    if requested_family != detected_family:
        raise ValueError(
            f"--estimator-family {requested_family} conflicts with checkpoint; "
            f"detected {detected_family} from intro_albedo.weight"
        )
    return requested_family


def read_checkpoint_feature_version(loadpath_network_g):
    """Read the feature contract without constructing a CUDA model."""
    if not loadpath_network_g or not os.path.isfile(loadpath_network_g):
        raise FileNotFoundError(f"network checkpoint does not exist: {loadpath_network_g}")
    checkpoint = torch.load(loadpath_network_g, map_location="cpu", weights_only=False)
    return checkpoint_feature_version(checkpoint)


def resolve_input_mode(requested_mode, family, feature_version):
    """Resolve capture encoding and reject an incompatible RAW checkpoint path."""
    aliases = {
        'auto': 'auto',
        'linear_rgb': 'linear_rgb',
        'gamma22': 'gamma22',
        'gamma22_png': 'gamma22',
    }
    try:
        mode = aliases[str(requested_mode).lower()] if requested_mode is not None else 'auto'
    except KeyError as error:
        raise ValueError(
            f"unsupported --input-encoding {requested_mode!r}; expected auto, linear_rgb, or gamma22_png"
        ) from error
    if mode == 'auto':
        mode = 'linear_rgb' if (
            family == 'matsynth21' and feature_version == 'raw_calibrated_v2'
        ) else 'gamma22'
    if feature_version == 'raw_calibrated_v2' and mode == 'gamma22':
        raise ValueError(
            "raw_calibrated_v2 checkpoints require --input-encoding linear_rgb; gamma22 is incompatible"
        )
    if family == 'legacy33' and mode == 'linear_rgb':
        raise ValueError("legacy33 inference requires the explicit gamma22 PNG input mode")
    return mode


def _load_legacy_model_class():
    """Import the denoising legacy path only when it is explicitly selected."""
    from model.nfplight_real_model import NFPLightModel

    return NFPLightModel


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_inference_manifest(args, family, model, input_mode=None, capture_records=None):
    """Record the selected model, input contract, geometry, and denoiser use."""
    checkpoint_path = os.path.abspath(args.loadpath_network_g)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if family == 'matsynth21':
        input_height, input_width = EXPECTED_INPUT_SHAPE[1:]
        network_name = 'TwoBranchNet'
    else:
        input_height = input_width = int(args.image_size)
        network_name = 'TwoBranchRealNet'
    feature_version = getattr(model, 'feature_version', 'legacy_batch_v0')
    normal_head = getattr(model, 'normal_head', 'xyz')
    if input_mode is None:
        input_mode = 'linear_rgb' if feature_version == 'raw_calibrated_v2' else 'gamma22'
    encoding = (
        'linear_rgb_float32_npy_or_uint16_png'
        if input_mode == 'linear_rgb' else 'gamma22_compat_png'
    )
    manifest = {
        'format': 'nfplight.real-inference.v1',
        'estimator_family': family,
        'network': network_name,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_intro_albedo_input_channels': CHECKPOINT_INPUT_CHANNELS[family],
        'strict_load': True,
        'feature_version': feature_version,
        'normal_head': normal_head,
        'input': {
            'source': 'RealDataset.inputs',
            'layout': 'near_rgb_then_far_rgb',
            'shape': f'[B,6,{input_height},{input_width}]',
            'dtype': 'float32',
            'mode': input_mode,
            'encoding': encoding,
            'range': [0.0, 1.0],
            'range_tolerance': INPUT_RANGE_TOLERANCE if family == 'matsynth21' else None,
            'alignment': 'caller-supplied',
        },
        'output': {
            'layout': 'NCHW',
            'shape': f'[B,{EXPECTED_OUTPUT_SHAPE[0]},{input_height},{input_width}]',
            'dtype': 'float32',
            'range': [-1.0, 1.0],
        },
        'feature_geometry': {
            'near_distance': float(model.near_distance),
            'far_distance': float(model.far_distance),
        },
        'denoiser': {
            'used': family == 'legacy33',
        },
        'limitations': [],
    }
    if capture_records is not None:
        manifest['captures'] = capture_records
    if family == 'legacy33':
        denoise_path = os.path.abspath(args.loadpath_network_denoise)
        manifest['denoiser'].update({
            'checkpoint': denoise_path,
            'checkpoint_sha256': sha256_file(denoise_path),
        })
    else:
        manifest['limitations'] = [
            'synthetic-to-real gap remains unvalidated',
            'feature geometry is training-compatible, not rig-calibrated',
            'near/far registration is caller-supplied',
        ]
    manifest_path = osp.join(args.save_root, 'inference_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path
    
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root',type=str,required=True,help="root path to save results.")
    parser.add_argument('--test_data_root',type=str,required=True,help="root path for data.")
    parser.add_argument('--loadpath_network_denoise',type=str,default=None,
                        help="legacy33 denoiser checkpoint; unused by matsynth21")
    parser.add_argument('--loadpath_network_g',type=str,required=True,
                        help="estimator checkpoint (required; no hard-coded default)")
    parser.add_argument('--estimator-family', choices=('auto','matsynth21','legacy33'),
                        default='auto',
                        help="auto-detect from intro_albedo.weight input channels")
    parser.add_argument('--image_size',type=int,default=256)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='inference device for matsynth21; auto uses CUDA when available')
    parser.add_argument('--input-encoding', choices=('auto', 'linear_rgb', 'gamma22_png'),
                        default=None,
                        help="capture encoding; auto selects linear_rgb for raw_calibrated_v2")
    
    args = parser.parse_args()

    makedirs(args.save_root)
    
    return args


def create_dataloader(args, input_mode=None):
    if input_mode is None:
        input_mode = getattr(args, 'input_mode', None)
    if input_mode is None:
        input_mode = getattr(args, 'input_encoding', 'auto')
    if input_mode == 'auto':
        # Direct callers without checkpoint context retain the old PNG behavior.
        input_mode = 'gamma22'
    elif input_mode == 'gamma22_png':
        input_mode = 'gamma22'
    dataset_opt = {
        'name': 'RealDataset',
        'root': args.test_data_root,
        'log': True,
        'input_mode': input_mode,
        'imageSize': args.image_size,   
    }
    test_set = RealDataset(dataset_opt)
    dataloader_opt = {
        'dataset': test_set,
        'batch_size': 1,
        'shuffle': False,
        'pin_memory': True,
        'num_workers': 1,
    }
    test_loader = torch.utils.data.DataLoader(**dataloader_opt)
    return test_loader

def test_pipeline(args):
    torch.backends.cudnn.benchmark = True

    family = resolve_estimator_family(
        getattr(args, 'estimator_family', 'auto'), args.loadpath_network_g
    )
    feature_version = read_checkpoint_feature_version(args.loadpath_network_g)
    input_mode = resolve_input_mode(
        getattr(args, 'input_mode', None) or getattr(args, 'input_encoding', 'auto'),
        family, feature_version
    )
    if family == 'legacy33':
        if getattr(args, 'device', 'auto') != 'auto':
            raise ValueError(
                'legacy33 inference uses its historical CUDA path; '
                '--device applies to matsynth21'
            )
        if not getattr(args, 'loadpath_network_denoise', None):
            raise ValueError(
                "legacy33 inference requires --loadpath_network_denoise"
            )
        model_class = _load_legacy_model_class()
    else:
        if getattr(args, 'loadpath_network_denoise', None):
            raise ValueError(
                "matsynth21 does not use DenoiseNet; omit --loadpath_network_denoise"
            )
        if int(args.image_size) != 256:
            raise ValueError("matsynth21 inference requires --image_size 256")
        model_class = MatSynthCaptureModel

    # create train and validation dataloaders
    # Keep the resolved mode on args for adapter validation and provenance.
    args.input_mode = input_mode
    test_loader = create_dataloader(args, input_mode=input_mode)

    # create model
    model = model_class(args)
    manifest_path = write_inference_manifest(args, family, model, input_mode=input_mode)
    print(f"inference manifest -> {manifest_path}")

    model.validation(test_loader)
    
        

if __name__ == '__main__':
    args = parse_options()
    
    test_pipeline(args)
