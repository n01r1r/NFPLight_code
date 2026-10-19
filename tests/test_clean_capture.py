import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np

import preprocess_dng as capture
from nfplight.registration import register_marker_frames_to_first
from nfplight.storage import identity, write_json

ROOT = Path(__file__).resolve().parents[1]


def geometry(offset = 0.):
    corners = {str(index): (np.array([[x, y], [x + 12, y], [x + 12, y + 12], [x, y + 12]],
                                     dtype = np.float32) + offset).tolist()
               for index, (x, y) in enumerate(((30, 30), (470, 30), (470, 470), (30, 470)))}
    return {'marker_ids': list(range(4)), 'marker_corners_raw': corners, 'homography': np.eye(3, dtype = np.float32).tolist()}


def test_registration_resolves_translation_in_common_plane():
    alignment = register_marker_frames_to_first([geometry(), geometry(2.)], max_rms = 1.)[-1]
    assert alignment['within_rms_limit']
    np.testing.assert_allclose(np.array(alignment['homography_final'])[:2, 2], [-2., -2.], atol = 2e-4)


@pytest.mark.parametrize('profile', ['capture_legacy', 'capture'])
def test_dng_prepare_review_finalize_roundtrip(tmp_path, monkeypatch, profile):
    source = tmp_path / 'source'
    source.mkdir()
    metadata = {side: {'filenames': [f'{side}_{index:02d}.dng' for index in range(5)], 'frameCount': 5, 'settings': {}}
                for side in ('near', 'far')}
    write_json(source / 'metadata.json', metadata)
    for section in metadata.values():
        for name in section['filenames']:
            (source / name).write_bytes(b'fixture')
    raw_metadata = {'raw_shape': [512, 512], 'cfa_pattern': [[0, 1], [1, 2]],
        'visible_cfa_indices': [[0, 1], [1, 2]], 'cfa_rgb_indices': [[0, 1], [1, 2]],
        'color_desc': 'RGBG', 'visible_origin': [0, 0], 'black_level_per_channel': [10] * 4,
        'white_level': 1010, 'camera_whitebalance': [1., 1., 1., 1.],
        'color_matrix': np.eye(3).tolist(), 'orientation_flip': 0}
    def unpack(path):
        counts = 110. if path.name.startswith('near') else 60.
        return SimpleNamespace(metadata = raw_metadata, rgb_float = np.full((512, 512, 3), counts, dtype = np.float32))
    monkeypatch.setattr(capture, 'unpack_dng', unpack)
    detected = geometry()
    detected.update(quad = [[0, 0], [511, 0], [511, 511], [0, 511]], detector_display_percentiles = [0., 1010.])
    monkeypatch.setattr(capture, 'detect_marker_geometry', lambda *args, **kwargs: (np.eye(3), detected))
    monkeypatch.setattr(capture, 'detect_target_geometry', lambda *args, **kwargs: (np.eye(3), detected))
    monkeypatch.setattr(capture, 'marker_record', lambda *args, **kwargs: (None, detected))
    monkeypatch.setattr(capture, 'read_dng_exposure_exif', lambda path: {'known': True, 'tags': {}, 'missing': []})
    monkeypatch.setattr(capture, 'read_color_transform', lambda path: {
        'normalized_rgb_wb': [1., 1., 1.], 'camera_to_linear_srgb': np.eye(3).tolist()})
    config = json.loads((ROOT / f'configs/{profile}.json').read_text())
    prepared = tmp_path / 'prepared'
    manifest = capture.prepare(source, prepared, config)
    assert manifest['status'] == 'awaiting_review'
    review = {'status': 'reviewed', 'texture_registration_passed': True, 'target_boundary_passed': True, 'note': 'fixture alignment',
             'preparation': identity(prepared / 'preparation.json')}
    if profile == 'capture':
        write_json(tmp_path / 'unreviewed.json', dict(review, target_boundary_passed = False))
        with pytest.raises(ValueError, match = 'boundary requires'):
            capture.finalize(prepared, tmp_path / 'unreviewed.json', tmp_path / 'unreviewed.pt')
    write_json(tmp_path / 'review.json', review)
    result = capture.finalize(prepared, tmp_path / 'review.json', tmp_path / 'pair.pt')
    assert result['pair'].shape == (1, 6, 256, 256) and result['pair'].dtype == torch.float32
    assert result['selected_indices'] == {'near': list(range(5)), 'far': list(range(5))}
    torch.testing.assert_close(result['pair'][:, :3], torch.full((1, 3, 256, 256), .1))
    torch.testing.assert_close(result['pair'][:, 3:], torch.full((1, 3, 256, 256), .05))
    stored = torch.load(tmp_path / 'pair.pt', weights_only = True)
    assert torch.equal(stored['pair'], result['pair'])
    with pytest.raises(FileExistsError):
        capture.finalize(prepared, tmp_path / 'review.json', tmp_path / 'pair.pt')
    (source / 'near_00.dng').write_bytes(b'changed source')
    with pytest.raises(ValueError, match = 'source changed'):
        capture.finalize(prepared, tmp_path / 'review.json', tmp_path / 'changed.pt')


def test_partial_tags_and_one_outlier_preserve_target_coordinates():
    from nfplight.registration import register_target_frame

    config = json.loads((ROOT / 'configs/capture.json').read_text())
    reference, current = geometry(), geometry(2.)
    current['marker_ids'].remove(3)
    del current['marker_corners_raw']['3']
    result = register_target_frame(reference, current, config)
    assert result['used_marker_ids'] == [0, 1, 2]
    np.testing.assert_allclose(np.array(result['homography_final'])[:2, 2], [-2., -2.], atol = 2e-3)
    current = geometry(2.)
    current['marker_corners_raw']['3'] = (np.array(current['marker_corners_raw']['3']) + 70).tolist()
    result = register_target_frame(reference, current, config)
    assert result['excluded_marker_ids'] == [3]
    np.testing.assert_allclose(np.array(result['homography_final'])[:2, 2], [-2., -2.], atol = 2e-3)
    current['marker_ids'] = [0, 1]
    with pytest.raises(ValueError):
        register_target_frame(reference, current, config)


def test_target_edges_include_rectangle_and_reject_blank_paper():
    from nfplight.target import target_quad

    config = json.loads((ROOT / 'configs/capture.json').read_text())['target']
    image = np.full((512, 512, 3), 210., dtype = np.float32)
    image[100:410, 95:405] = [50., 80., 100.]
    image[160:190, 95:405] = [90., 130., 160.]
    markers = geometry()['marker_corners_raw']
    quad, evidence = target_quad(image, markers, config)
    np.testing.assert_allclose(quad, [[96, 101], [403, 101], [403, 408], [96, 408]], atol = 3)
    assert min(evidence['edge_support']) >= config['min_edge_support']
    with pytest.raises(ValueError, match = 'target boundary'):
        target_quad(np.full_like(image, 210), markers, config)


def test_invalid_target_settings_and_clustered_tags_are_rejected():
    from nfplight.target import validate_target_config
    from nfplight.registration import register_target_frame

    config = json.loads((ROOT / 'configs/capture.json').read_text())
    config['target']['canny_low'] = float('nan')
    with pytest.raises(ValueError, match = 'finite'):
        validate_target_config(config)
    config = json.loads((ROOT / 'configs/capture.json').read_text())
    record = geometry()
    for key, points in record['marker_corners_raw'].items():
        record['marker_corners_raw'][key] = (np.array(points) * .01).tolist()
    with pytest.raises(ValueError, match = 'span'):
        register_target_frame(record, record, config)

    diagonal = geometry()
    for index in range(4):
        base = 40 + index * 140
        diagonal['marker_corners_raw'][str(index)] = [[base, base], [base + 12, base], [base + 12, base + 12], [base, base + 12]]
    with pytest.raises(ValueError, match = 'footprint'):
        register_target_frame(diagonal, diagonal, config)
