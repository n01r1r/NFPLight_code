import hashlib
import json
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from capture_processing import memory_refresh


@pytest.mark.parametrize("failures,expected_value,count", [([], 0.3, 5), (["far_01 RMS failed"], 0.1, 1)])
def test_memory_inference_reviews_population_without_npz_or_sha(tmp_path, monkeypatch, failures, expected_value, count):
    output = tmp_path / "sample" / "refresh"
    output.mkdir(parents=True)
    (output / "root_review.json").write_text(json.dumps({
        "status": "reviewed", "texture_registration_passed": True}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("NPZ saving and SHA-256 are forbidden in memory refresh")

    monkeypatch.setattr(np, "savez", forbidden)
    monkeypatch.setattr(np, "savez_compressed", forbidden)
    monkeypatch.setattr(hashlib, "sha256", forbidden)
    for name in ("_save_display_previews", "_save_prediction_component_previews", "_render_array"):
        monkeypatch.setattr(memory_refresh, name, lambda *args, **kwargs: [])
    monkeypatch.setattr(memory_refresh, "_figure_gallery", lambda *args, **kwargs: [])
    monkeypatch.setattr(memory_refresh, "_make_html", lambda *args: '<meta charset="utf-8"><main>NPZ</main>')
    monkeypatch.setattr(memory_refresh, "publish_memory_records", lambda *args: None)
    transform = {"normalized_rgb_wb": [1, 1, 1], "camera_to_linear_srgb": np.eye(3).tolist()}
    metadata = {"black_level_per_channel": [0, 0, 0, 0], "white_level": 100}
    counts = [np.full((2, 2, 3), 10 * (index + 1), dtype=np.float32) for index in range(5)]
    model = SimpleNamespace(device=torch.device("cpu"), net_g=torch.nn.Identity())

    def infer(tensor):
        np.testing.assert_allclose(tensor.numpy(), expected_value)
        features = torch.zeros((1, 33, 2, 2), dtype=torch.float32)
        model.last_estimator_input = features
        prediction = torch.zeros((1, 10, 2, 2), dtype=torch.float32)
        return prediction, {"features_legacy33": features}

    model.infer = infer
    group = {"output": output, "archive": tmp_path, "failures": failures, "protected": {},
             "counts": {side: counts for side in ("near", "far")},
             "previews": {side: [value / 100 for value in counts] for side in ("near", "far")},
             "metadata": {side: {"metadata": [metadata] * 5} for side in ("near", "far")},
             "manifest": {"source": "source", "capture_metadata": {}, "model_geometry": {},
                          "model_contract": {"estimator_family": "author_legacy33", "feature_channels": 33},
                          "color_transforms": {side: transform for side in ("near", "far")}}}
    record = memory_refresh.infer_memory_group(group, model)
    assert record["frame_average"]["enabled"] is (count == 5)
    assert record["selected_frames"]["selected_indices"] == {"near": list(range(count)), "far": list(range(count))}
    assert not list(tmp_path.rglob("*.npz"))
    assert 'base href="refresh/"' in (output.parent / "report.html").read_text(encoding="utf-8")


def test_memory_inference_refuses_unreviewed_frames(tmp_path):
    (tmp_path / "root_review.json").write_text('{"status":"pending"}', encoding="utf-8")
    with pytest.raises(ValueError, match="actual visual review"):
        memory_refresh.infer_memory_group({"output": tmp_path}, None)


def test_publication_preserves_previous_json_without_npz(tmp_path):
    output = tmp_path / "sample" / "refresh"
    output.mkdir(parents=True)
    (output.parent / "manifest.json").write_text('{"version":"previous"}', encoding="utf-8")
    (output / "registration.json").write_text('{"reference":"near_00"}', encoding="utf-8")
    manifest = {"version": "current", "root_visual_review": {"status": "reviewed"},
                "verification": {"npz_saved": False}}
    memory_refresh.publish_memory_records(output, manifest)
    assert json.loads((output.parent / "manifest.json").read_text(encoding="utf-8")) == manifest
    history = output.parent / "history_before_refresh_20261006" / "manifest.json"
    assert json.loads(history.read_text(encoding="utf-8")) == {"version": "previous"}
    assert not list(tmp_path.rglob("*.npz"))


def test_memory_preparation_uses_crop_only_policy(tmp_path, monkeypatch):
    archive=tmp_path / "archive"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({"color_transforms":{"near":{},"far":{}}}),encoding="utf8")
    metadata={"metadata":[{"capture_settings":{},"dng_exif":{}} for _ in range(5)]}
    (archive / "preparation.json").write_text(json.dumps({"sides":{"near":metadata,"far":metadata}}),encoding="utf8")
    image=np.full((512,512,3),7,dtype=np.float32)
    image[46:466,46:466]=np.float32(0.25)
    class Saved:
        def __enter__(self): return {"demosaic_rgb_selected_dtype":image}
        def __exit__(self,*args): pass
    monkeypatch.setattr(np,"load",lambda *a,**k:Saved())
    from pathlib import Path
    real_stat=Path.stat
    monkeypatch.setattr(Path,"stat",lambda p,*a,**k:SimpleNamespace(st_size=123,st_mtime_ns=456) if p.suffix==".npz" else real_stat(p,*a,**k))
    def forbidden(*args,**kwargs): raise AssertionError("No NPZ save or SHA allowed")
    monkeypatch.setattr(np,"savez",forbidden)
    monkeypatch.setattr(np,"savez_compressed",forbidden)
    monkeypatch.setattr(hashlib,"sha256",forbidden)
    def detect(rgb,*,inset):
        assert inset==0.0
        return np.eye(3,dtype=np.float32),{"inset":0.0}
    def register(geometries,*,crop_margin):
        assert crop_margin==46
        return [{"homography_final":np.eye(3,dtype=np.float32).tolist(),"within_rms_limit":True,"residual_rms_px":0.0} for _ in geometries]
    monkeypatch.setattr(memory_refresh,"detect_marker_geometry",detect)
    monkeypatch.setattr(memory_refresh,"register_marker_frames_to_first",register)
    monkeypatch.setattr(memory_refresh,"_linear",lambda a,*args:(a,a,a,a))
    monkeypatch.setattr(memory_refresh,"_contact",lambda *args:None)
    monkeypatch.setattr(memory_refresh,"_render_array",lambda *args,**kwargs:None)
    import capture_processing.burst as burst
    monkeypatch.setattr(burst,"_metadata_signature",lambda *args:{})
    monkeypatch.setattr(memory_refresh,"validate_burst_metadata",lambda *args:None)
    output=tmp_path / "prepared"
    group=memory_refresh.prepare_memory_group(archive,output)
    for side in ("near","far"):
        assert len(group["counts"][side])==5
        for counts in group["counts"][side]:
            assert counts.dtype==np.float32
            np.testing.assert_allclose(counts,0.25,rtol=0,atol=1e-7)
            from capture_processing.geometry import rectify_stages
            np.testing.assert_array_equal(counts,rectify_stages(image,np.eye(3,dtype=np.float32))["output256"])
    policy=json.loads((output / "registration.json").read_text(encoding="utf8"))["spatial_policy"]
    assert policy["version"]=="sample_crop_only_v1"
    assert policy["crop_margin"]==46 and policy["sample_side_cm"]==5
    assert not list(tmp_path.rglob("*.npz"))
