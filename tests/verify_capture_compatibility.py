import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nfplight.storage import identity, write_json
from preprocess_dng import prepare
from nfplight.registration import register_target_frame, _project_points


def partial_tag_check(reference, config):
    centers = {tag: np.mean(reference['marker_corners_raw'][str(tag)], axis = 0) for tag in reference['marker_ids']}
    def area(tags):
        a, b = centers[tags[1]] - centers[tags[0]], centers[tags[2]] - centers[tags[0]]
        return abs(a[0] * b[1] - a[1] * b[0])
    chosen = max(combinations(reference['marker_ids'], 3), key = area)
    shift = np.asarray([17, -9], dtype = np.float32)
    current = {'marker_ids': list(chosen), 'homography': np.eye(3, dtype = np.float32).tolist(),
        'marker_corners_raw': {str(tag): (np.asarray(reference['marker_corners_raw'][str(tag)], dtype = np.float32) + shift).tolist() for tag in chosen}}
    result = register_target_frame(reference, current, config)
    quad = np.asarray(reference['quad'], dtype = np.float32)
    actual = _project_points(np.asarray(result['homography_final'], dtype = np.float32), quad + shift)
    expected = _project_points(np.asarray(reference['homography'], dtype = np.float32), quad)
    error = float(np.max(np.abs(actual - expected)))
    if error > .01 or len(result['used_marker_ids']) != 3:
        raise ValueError('synthetic partial-tag translation recovery failed')
    return {'remaining_tags': list(chosen), 'max_warp_pixel_error': error,
        'scope': 'synthetic dropout/translation of real coordinates, not a new occlusion capture'}


def verify(args):
    config = json.loads(Path(args.config).read_text(encoding = 'utf-8-sig'))
    rows = []
    for source in sorted(Path(args.source_root).glob('26*')):
        prepared = Path(args.prepared_root) / source.name
        manifest_path = prepared / 'preparation.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else prepare(source, prepared, config)
        recorded = dict(manifest['config'])
        recorded.setdefault('spatial_policy', 'legacy_crop46')
        if recorded != config or manifest['status'] != 'awaiting_review':
            raise ValueError('incomplete preparation or different capture config')
        for item in manifest['source_files']:
            if identity(item['path']) != item:
                raise ValueError('capture source identity changed')
        baseline = None
        if args.reference_registration:
            baseline = json.loads((Path(args.reference_registration) / source.name / 'preparation/registration.json').read_text())
        errors, eligible, used, excluded = [], [], [], []
        for side in ('near', 'far'):
            if len(manifest['frames'][side]) != 5:
                raise ValueError('capture must retain five frame records')
            for index, frame in enumerate(manifest['frames'][side]):
                if frame['error'] is not None:
                    errors.append({'frame': frame['filename'], 'error': frame['error']})
                    continue
                if identity(frame['archive']['path']) != frame['archive']:
                    raise ValueError('prepared count identity changed')
                with np.load(frame['archive']['path'], allow_pickle = False) as archive:
                    counts = archive['counts']
                    if counts.dtype != np.float32 or counts.shape != (256, 256, 3) or not np.isfinite(counts).all():
                        raise ValueError('invalid FP32 prepared counts')
                if config['spatial_policy'] == 'robust_target':
                    recomputed = register_target_frame(manifest['frames']['near'][0]['geometry'], frame['geometry'], config)
                    for key in ('homography_final', 'used_marker_ids', 'within_rms_limit'):
                        if recomputed[key] != frame['alignment'][key]:
                            raise ValueError(f'current registration differs from prepared frame: {key}')
                if baseline is not None:
                    original = baseline['frames'][side][index]
                    if (original['geometry'] != frame['geometry'] or original['alignment'] != frame['alignment']
                            or original['full_crop_support'] != frame['full_crop']):
                        raise ValueError('legacy registration or crop support differs')
                eligible.append(frame['eligible'])
                used.append(len(frame['alignment'].get('used_marker_ids', frame['alignment']['shared_marker_ids'])))
                excluded.append(frame['alignment'].get('excluded_marker_ids', []))
        usable = all(manifest['frames'][side][0].get('full_crop') and manifest['frames'][side][0]['error'] is None for side in ('near', 'far'))
        partial = partial_tag_check(manifest['frames']['near'][0]['geometry'], config) if config['spatial_policy'] == 'robust_target' else None
        rows.append({'group': source.name, 'partial_tag_check': partial, 'errors': errors, 'eligible_count': sum(eligible),
            'frame00_usable': bool(usable), 'used_tag_counts': used, 'excluded_tags': excluded,
            'preparation': identity(manifest_path), 'legacy_geometry_exact': baseline is not None})
        print(source.name, 'errors', len(errors), 'eligible', sum(eligible), flush = True)
    passed = len(rows) == 5 and all(not row['errors'] and row['frame00_usable'] for row in rows)
    write_json(args.out, {'passed': passed, 'groups': rows, 'spatial_policy': config['spatial_policy'],
        'human_texture_and_target_review_performed': False, 'sha256_computed': False})
    if not passed:
        raise ValueError('capture verification failed; inspect report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Verify all five real DNG preparations without approving human review')
    parser.add_argument('--source-root', required = True)
    parser.add_argument('--prepared-root', required = True)
    parser.add_argument('--config', default = str(ROOT / 'configs/capture.json'))
    parser.add_argument('--reference-registration')
    parser.add_argument('--out', required = True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError('verification output already exists')
    verify(args)
