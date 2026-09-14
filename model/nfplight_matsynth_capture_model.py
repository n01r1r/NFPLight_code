"""MatSynth 21-channel estimator adapter for linear real captures.

The synthetic estimator's feature assembly is kept in
``model.nfplight_model.NFPLightModel.test``.  This adapter only validates the
RealDataset input contract and supplies a strict, normalized checkpoint load;
it never constructs a denoising network.
"""

from copy import deepcopy
from os import path as osp

import torch
from tqdm import tqdm

from model.nfplight_model import NFPLightModel as MatSynthBaseModel
from model.nfplight_model import checkpoint_feature_version
from model.normal_head import apply_normal_head, checkpoint_normal_head
from network.nfplight_net import TwoBranchNet


EXPECTED_INPUT_SHAPE = (6, 256, 256)
EXPECTED_OUTPUT_SHAPE = (10, 256, 256)
INPUT_RANGE_TOLERANCE = 1e-5
CHECKPOINT_WRAPPER_KEYS = ("params", "state_dict")
CHECKPOINT_PREFIXES = {"module", "_orig_mod"}


def _uncollate(value):
    """Turn a batch-size-one DataLoader value back into JSON-friendly values."""
    if isinstance(value, dict):
        return {key: _uncollate(item) for key, item in value.items()}
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _uncollate(value[0])
        return [_uncollate(item) for item in value]
    return value


def checkpoint_state_dict(checkpoint):
    """Extract a parameter mapping from a supported checkpoint payload."""
    if not isinstance(checkpoint, dict):
        raise ValueError("network checkpoint must contain a parameter mapping")
    for wrapper_key in CHECKPOINT_WRAPPER_KEYS:
        wrapped = checkpoint.get(wrapper_key)
        if isinstance(wrapped, dict):
            return wrapped
    return checkpoint


def normalize_checkpoint_state_dict(checkpoint):
    """Normalize DataParallel/compiled prefixes without relaxing strict load.

    ``module`` and ``_orig_mod`` are wrapper segments, not estimator layer
    names.  Duplicate normalized keys are rejected because silently choosing
    one would make checkpoint provenance ambiguous.
    """
    raw_state_dict = checkpoint_state_dict(checkpoint)
    normalized = {}
    for raw_key, value in raw_state_dict.items():
        if not isinstance(raw_key, str):
            raise ValueError("network checkpoint parameter keys must be strings")
        key_parts = [part for part in raw_key.split(".") if part not in CHECKPOINT_PREFIXES]
        normalized_key = ".".join(key_parts)
        if not normalized_key:
            raise ValueError(f"invalid empty normalized checkpoint key: {raw_key}")
        if normalized_key in normalized:
            raise ValueError(
                f"duplicate checkpoint key after normalization: {normalized_key}"
            )
        normalized[normalized_key] = value
    return normalized


class MatSynthCaptureModel(MatSynthBaseModel):
    """Run the synthetic 21-channel estimator on linear real captures.

    The inherited ``test`` method remains authoritative for the 21-channel
    relation-map assembly and synthetic geometry (near=2.414, far=10).  The
    input contract is ``[B,6,256,256]`` linear values in ``[0,1]`` up to the
    named tolerance.  No DenoiseNet is constructed or consulted.
    """

    def __init__(self, args):
        if getattr(args, "image_size", None) != 256:
            raise ValueError("matsynth21 inference requires --image_size 256")
        super().__init__(args)

    def init_network(self):
        """Create and strictly load the 21-channel network."""
        self.net_g = TwoBranchNet()
        self.net_g = self.model_to_device(self.net_g)
        self.load_network(self.net_g, self.args.loadpath_network_g)

    def load_network(self, net, load_path, strict=True, param_key="params", IsCompile=False):
        """Load normalized checkpoint keys with strict architecture matching."""
        if not strict:
            raise ValueError("MatSynthCaptureModel requires strict checkpoint loading")
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        self.feature_version = checkpoint_feature_version(checkpoint)
        self.normal_head = checkpoint_normal_head(checkpoint)
        requested_mode = getattr(self.args, "input_mode", None)
        if requested_mode is None:
            requested_mode = getattr(self.args, "input_encoding", None)
        if self.feature_version == "raw_calibrated_v2" and str(requested_mode).lower() in {
            "gamma22", "gamma22_png", "gamma22_compat", "legacy_png", "encoded_png"
        }:
            raise ValueError(
                "raw_calibrated_v2 checkpoints require linear_rgb input; gamma22 is incompatible"
            )
        state_dict = normalize_checkpoint_state_dict(checkpoint)
        net.load_state_dict(deepcopy(state_dict), strict=True)

    def test(self):
        """Run feature assembly and the checkpoint-selected fixed decoder."""
        self.net_g.eval()
        with torch.no_grad():
            inputs, self.relation_map, self.aligned_far_img = self.build_features(
                self.inputs
            )
            raw_prediction = self.net_g(inputs)
            self.pred_svbrdf = apply_normal_head(
                raw_prediction, normal_head=getattr(self, "normal_head", "xyz")
            )

    def feed_data(self, data, random=True):
        """Accept a RealDataset batch without gamma conversion or denoising."""
        if "inputs" not in data:
            raise KeyError("RealDataset batch must contain 'inputs'")
        inputs = data["inputs"]
        if not torch.is_tensor(inputs):
            raise TypeError("RealDataset 'inputs' must be a torch.Tensor")
        if inputs.ndim != 4:
            raise ValueError(
                "matsynth21 expects rank-4 inputs [B,6,256,256], "
                f"got rank {inputs.ndim} with shape {tuple(inputs.shape)}"
            )
        if not inputs.is_floating_point():
            raise TypeError("RealDataset 'inputs' must be a floating-point linear RGB tensor")
        if inputs.shape[0] <= 0 or tuple(inputs.shape[1:]) != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                "matsynth21 expects inputs shaped [B, 6, 256, 256], "
                f"got {tuple(inputs.shape)}"
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("RealDataset 'inputs' contains non-finite values")
        min_value = float(inputs.amin().item())
        max_value = float(inputs.amax().item())
        if min_value < -INPUT_RANGE_TOLERANCE or max_value > 1.0 + INPUT_RANGE_TOLERANCE:
            raise ValueError(
                "matsynth21 expects linear inputs in [0,1] within tolerance "
                f"{INPUT_RANGE_TOLERANCE:g}; observed [{min_value:.6g}, {max_value:.6g}]"
            )
        # Values just outside the interval can occur after image decoding due
        # to floating-point round-off.  The training renderer clips to this
        # interval, so apply the same contract before logarithmic features are
        # assembled while still rejecting materially invalid captures above.
        self.inputs = inputs.to(dtype=torch.float32).clamp(0.0, 1.0).to(self.device)
        self.name = data.get("name", ["capture"])
        self.capture_metadata = _uncollate(data.get("metadata", {}))
        self.input_hashes = _uncollate(data.get("input_hashes", {}))

    def validation(self, dataloader):
        """Save predictions while preserving RealDataset directory names."""
        pbar = tqdm(total=len(dataloader), unit="image")
        self.metric_results = 0
        for val_data in dataloader:
            self.feed_data(val_data, random=False)
            self.test()
            img_name = _uncollate(self.name)
            if not isinstance(img_name, str):
                img_name = str(img_name)
            output_path = osp.join(self.args.save_root, img_name)
            input_mode = getattr(self.args, "input_mode", None)
            if input_mode is None:
                input_mode = getattr(self.args, "input_encoding", "linear_rgb")
            provenance = {
                "capture_name": img_name,
                "input_mode": input_mode,
                "capture_metadata": self.capture_metadata,
                "input_hashes": self.input_hashes,
            }
            self.save_svbrdf_maps(self.pred_svbrdf, output_path, metadata=provenance)
            # PNG remains a preview; authoritative maps are the float NPZ above.
            self.save_svbrdf(self.pred_svbrdf, output_path, gamma=True)
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description("Testing")
        pbar.close()
