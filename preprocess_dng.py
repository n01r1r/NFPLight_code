import argparse
import json
from pathlib import Path

import torch

from einops import rearrange
import numpy as np
from PIL import Image, ImageDraw

from nfplight.geometry import area_resize, detect_marker_geometry, warp_perspective
from nfplight.photometry import apply_compensation, apply_linear_color, read_color_transform
from nfplight.raw import unpack_dng
from nfplight.target import detect_target_geometry, marker_record, validate_target_config
from nfplight.registration import (register_marker_frames_to_first, validate_burst_metadata,
    read_dng_exposure_exif, _metadata_signature, _safe_burst_filename, register_target_frame)
from nfplight.storage import identity, write_json, atomic_write, output_lock


def read_json(path):
    return json.loads(Path(path).read_text(encoding = 'utf-8-sig'))


def prepare(source, output, config):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('output cannot be inside the source capture')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('preparation requires a new output directory')
    if config.get('spatial_policy', 'legacy_crop46') not in ('legacy_crop46', 'robust_target'):
        raise ValueError('unknown capture spatial policy')
    robust = config.get('spatial_policy') == 'robust_target'
    if robust:
        validate_target_config(config)
    expected_margin = 0 if robust else 46
    if (not np.isfinite(config['max_marker_rms_px']) or config['max_marker_rms_px'] != 1.
            or config['sample_side_cm'] != 5. or config['photometry'] != 'both' or config['inset'] != 0 or config['burst_count'] != 5
            or (config['warp_size'], config['crop_margin'], config['output_size']) != (512, expected_margin, 256)):
        raise ValueError('capture contract requires both, five frames and declared 512px target policy')
    metadata = read_json(source / 'metadata.json')
    paths = {}
    for side in ('near', 'far'):
        section = metadata[side]
        names = section['filenames']
        if len(names) != config['burst_count'] or section.get('frameCount', 5) != 5 or len(set(names)) != 5:
            raise ValueError('metadata requires five distinct frames on each side')
        paths[side] = [source / _safe_burst_filename(name, side) for name in names]
        if any(path.resolve().parent != source or not path.is_file() for path in paths[side]):
            raise ValueError('frame missing or outside capture directory')
    originals = [identity(source / 'metadata.json')] + [identity(path) for values in paths.values() for path in values]
    output.mkdir(parents = True, exist_ok = True)
    manifest = {'schema': 'nfplight.prepared.v1', 'source': str(source), 'config': config,
                'source_files': originals, 'frames': {'near': [], 'far': []}, 'status': 'preparing',
                'identity_method': 'path_size_mtime_ns; not a cryptographic content check', 'sha256_computed': False}
    reference = None
    signatures = {}
    with output_lock(output):
        for side in ('near', 'far'):
            signatures[side] = None
            for index, path in enumerate(paths[side]):
                row = {'index': index, 'filename': path.name, 'error': None, 'eligible': False}
                manifest['frames'][side].append(row)
                try:
                    capture = unpack_dng(path)
                    black = capture.metadata['black_level_per_channel']
                    if not black or len(set(black)) != 1 or not 0 <= black[0] < capture.metadata['white_level']:
                        raise ValueError('black/white calibration must be valid and equal-channel')
                    exposure = read_dng_exposure_exif(path)
                    signature = _metadata_signature(capture.metadata, metadata[side].get('settings'), exposure)
                    if index == 0:
                        signatures[side] = signature
                    metadata_error = None
                    try:
                        validate_burst_metadata(signatures[side], signature)
                    except (ValueError, TypeError) as error:
                        metadata_error = str(error)
                    if robust:
                        if side == 'near' and index == 0:
                            _, geometry = detect_target_geometry(capture.rgb_float, config)
                        else:
                            _, geometry = marker_record(capture.rgb_float, min_tags = config['registration']['min_shared_tags'])
                    else:
                        _, geometry = detect_marker_geometry(capture.rgb_float, inset = config['inset'])
                    if side == 'near' and index == 0:
                        reference = geometry
                        if robust:
                            low, high = geometry['detector_display_percentiles']
                            display = np.uint8(np.clip((capture.rgb_float - low) / max(high - low, 1.), 0, 1) * 255 + .5)
                            preview = Image.fromarray(display)
                            preview.thumbnail((1024, 1024))
                            factor = preview.width / capture.rgb_float.shape[1]
                            points = [(x * factor, y * factor) for x, y in geometry['quad']]
                            ImageDraw.Draw(preview).line(points + [points[0]], fill = (255, 0, 0), width = 3)
                            preview.save(output / 'target_boundary.png')
                            manifest['target_boundary_preview'] = identity(output / 'target_boundary.png')
                    if reference is None:
                        raise ValueError('near_00 common reference unavailable')
                    if robust:
                        alignment = register_target_frame(reference, geometry, config)
                    else:
                        alignment = register_marker_frames_to_first([reference, geometry],
                            warp_size = config['warp_size'], crop_margin = config['crop_margin'],
                            output_size = config['output_size'], max_rms = config['max_marker_rms_px'])[-1]
                        if side == 'near' and index == 0:
                            alignment = register_marker_frames_to_first([reference], max_rms = config['max_marker_rms_px'])[0]
                    matrix = np.asarray(alignment['homography_final'], dtype = np.float32)
                    warped, valid = warp_perspective(capture.rgb_float, matrix,
                        (config['warp_size'], config['warp_size']), return_valid = True)
                    margin = config['crop_margin']
                    region = slice(margin, -margin if margin else None)
                    crop, crop_valid = warped[region, region], valid[region, region]
                    counts = area_resize(crop, (config['output_size'], config['output_size']))
                    archive = output / f'{side}_{index:02d}.npz'
                    atomic_write(archive, lambda stream: np.savez(stream, counts = counts, valid_crop = crop_valid))
                    # Preview pixels are review-only and never read as model inputs.
                    preview = np.uint8(np.clip(counts / capture.metadata['white_level'], 0, 1) * 255 + .5)
                    preview_path = output / f'{side}_{index:02d}.png'
                    Image.fromarray(preview).save(preview_path)
                    row.update(archive = identity(archive), preview = identity(preview_path),
                        metadata = capture.metadata, alignment = alignment, geometry = geometry,
                        full_crop = bool(crop_valid.all()), metadata_error = metadata_error,
                        color = read_color_transform(path), exposure = exposure,
                        eligible = bool(alignment['within_rms_limit'] and crop_valid.all() and exposure['known'] and metadata_error is None))
                except Exception as error:
                    row['error'] = f'{type(error).__name__}: {error}'
                write_json(output / 'preparation.json', manifest)
        if [identity(item['path']) for item in originals] != originals:
            raise ValueError('source changed during preparation')
        if any(manifest['frames'][side][0]['error'] is not None or not manifest['frames'][side][0].get('full_crop') for side in ('near', 'far')):
            manifest['status'] = 'rejected'
            write_json(output / 'preparation.json', manifest)
            raise ValueError('near_00/far_00 cannot supply a valid target; inspect preparation errors')
        manifest['status'] = 'awaiting_review'
        write_json(output / 'preparation.json', manifest)
        write_json(output / 'review.example.json', {'status': 'pending', 'texture_registration_passed': None,
            'target_boundary_passed': None if robust else True,
            'note': '', 'preparation': identity(output / 'preparation.json')})
    return manifest


def select_frames(manifest, review):
    if (review.get('status') != 'reviewed' or type(review.get('texture_registration_passed')) is not bool
            or not isinstance(review.get('note'), str) or not review['note'].strip()):
        raise ValueError('review requires reviewed status, boolean texture decision and nonempty note')
    reasons = []
    for side in ('near', 'far'):
        frames = manifest['frames'][side]
        if len(frames) != 5 or [row['index'] for row in frames] != list(range(5)):
            raise ValueError('prepared burst must retain all five frame records')
        first = frames[0]
        if first['error'] is not None or first.get('full_crop') is not True:
            raise ValueError('frame00 is not a usable full-crop fallback')
        reasons.extend(f'{side}_{row["index"]:02d}: quantitative gate failed' for row in frames if not row['eligible'])
    if not review['texture_registration_passed']:
        reasons.append('texture registration review failed')
    return ([0] if reasons else list(range(5))), reasons


def finalize(prepared, review_path, output):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('input pair output already exists')
    manifest = read_json(prepared / 'preparation.json')
    review = read_json(review_path)
    if review.get('preparation') != identity(prepared / 'preparation.json'):
        raise ValueError('review must identify the exact preparation file')
    for item in manifest['source_files']:
        if identity(item['path']) != item:
            raise ValueError('source changed since preparation')
    if manifest['config'].get('spatial_policy') == 'robust_target':
        if review.get('target_boundary_passed') is not True:
            raise ValueError('robust target boundary requires explicit passed review')
        if identity(manifest['target_boundary_preview']['path']) != manifest['target_boundary_preview']:
            raise ValueError('target boundary preview changed')
    indices, reasons = select_frames(manifest, review)
    crop_size = manifest['config']['warp_size'] - 2 * manifest['config']['crop_margin']
    arrays, unclipped = [], []
    for side in ('near', 'far'):
        counts = None
        for index in indices:
            row = manifest['frames'][side][index]
            item = row['archive']
            path = Path(item['path']).resolve()
            if path.parent != prepared or identity(path) != item:
                raise ValueError('prepared frame identity changed or escapes output')
            if identity(row['preview']['path']) != row['preview']:
                raise ValueError('review preview changed')
            with np.load(path, allow_pickle = False) as archive:
                value, valid = archive['counts'], archive['valid_crop']
                if value.shape != (256, 256, 3) or value.dtype != np.float32 or not np.isfinite(value).all() or valid.shape != (crop_size, crop_size) or valid.dtype != np.bool_ or not valid.all():
                    raise ValueError('invalid prepared count array or crop support')
                if counts is None:
                    counts = value.copy()
                else:
                    np.add(counts, value, out = counts)
        counts /= np.float32(len(indices))
        first = manifest['frames'][side][0]
        meta = first['metadata']
        sensor = apply_compensation(counts, meta['black_level_per_channel'][0], meta['white_level'], 'both')
        _, linear = apply_linear_color(sensor, first['color'])
        unclipped.append(linear)
        arrays.append(np.clip(linear, 0, 1))
    pair = rearrange(torch.from_numpy(np.concatenate(arrays, axis = -1)), 'h w c -> 1 c h w')
    payload = {'schema': 'nfplight.input.v1', 'pair': pair,
        'valid_mask': torch.ones(1, 1, 256, 256, dtype = torch.bool),
        'unclipped_pair': rearrange(torch.from_numpy(np.concatenate(unclipped, axis = -1)), 'h w c -> 1 c h w'),
        'capture_gain_mode': 'author_patch20', 'spatial_policy': manifest['config'].get('spatial_policy', 'legacy_crop46'), 'selected_indices': {'near': indices, 'far': indices},
        'fallback_reasons': reasons, 'preparation': manifest, 'review': review}
    atomic_write(output, lambda stream: torch.save(payload, stream))
    return payload


def main():
    parser = argparse.ArgumentParser(description = 'DNG preparation and reviewed FP32 near/far export')
    commands = parser.add_subparsers(dest = 'command', required = True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--source', required = True)
    prepare_parser.add_argument('--out', required = True)
    prepare_parser.add_argument('--config', default = str(Path(__file__).parent / 'configs/capture.json'))
    finalize_parser = commands.add_parser('finalize')
    finalize_parser.add_argument('--prepared', required = True)
    finalize_parser.add_argument('--review', required = True)
    finalize_parser.add_argument('--out', required = True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source, args.out, read_json(args.config))
    else:
        finalize(args.prepared, args.review, args.out)


if __name__ == '__main__':
    main()
