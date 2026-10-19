import cv2
import numpy as np

from nfplight.geometry import detect_marker_observations, _perspective_from_quad, _single_solve


def validate_target_config(config):
    target, registration = config['target'], config['registration']
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in (*target.values(), *registration.values())):
        raise ValueError('target and registration settings must be finite numbers')
    for key in ('detection_size', 'blur_size', 'close_size', 'edge_tolerance_px', 'line_votes', 'line_gap_px'):
        if type(target[key]) is not int or target[key] < 1:
            raise ValueError(f'target requires positive integer {key}')
    if (not 0 < target['min_area_fraction'] < target['max_area_fraction'] < 1
            or not 0 < target['canny_low'] < target['canny_high'] <= 255
            or not 0 < target['polygon_epsilon'] < .1 or target['inset_pixels'] < 0
            or target['line_band_px'] <= 0 or target['line_distance_weight'] < 0
            or any(not 0 < target[key] <= 1 for key in ('min_rectangularity', 'min_edge_support', 'min_line_fraction', 'min_line_cosine'))
            or type(registration['min_shared_tags']) is not int or registration['min_shared_tags'] < 3
            or type(registration['max_trials']) is not int or registration['max_trials'] < 1
            or not 0 < registration['min_axis_span'] <= 1
            or not 0 <= registration['max_rejected_fraction'] <= .5):
        raise ValueError('invalid target or registration acceptance settings')


def marker_record(rgb, *, min_tags):
    source, corners, ids, low, high = detect_marker_observations(rgb, min_tags = min_tags)
    tags = {str(int(tag)): np.asarray(points, dtype = np.float32).reshape(4, 2).tolist()
            for tag, points in zip(ids.reshape(-1), corners)}
    return source, {'marker_ids': [int(tag) for tag in ids.reshape(-1)],
        'marker_corners_raw': tags, 'tags': len(tags), 'homography': np.eye(3, dtype = np.float32).tolist(),
        'detector_display_percentiles': [float(low), float(high)]}


def refine_quad(edges, points, config):
    lines = []
    for a, b in zip(points, np.roll(points, -1, axis = 0)):
        length = np.linalg.norm(b - a)
        direction = (b - a) / length
        normal = np.asarray([-direction[1], direction[0]], dtype = np.float32)
        segments = cv2.HoughLinesP(edges, 1, np.pi / 180, config['line_votes'],
            minLineLength = length * config['min_line_fraction'], maxLineGap = config['line_gap_px'])
        candidates = []
        if segments is not None:
            for segment in segments[:, 0]:
                first, second = segment.reshape(2, 2).astype(np.float32)
                delta = second - first
                span = np.linalg.norm(delta)
                alignment = abs(float(delta @ direction / span))
                distance = float(np.abs((np.stack((first, second)) - a) @ normal).mean())
                along = float(((first + second) * .5 - a) @ direction / length)
                if alignment >= config['min_line_cosine'] and distance <= config['line_band_px'] and 0 <= along <= 1:
                    candidates.append((span - config['line_distance_weight'] * distance, first, second))
        if not candidates:
            raise ValueError('target boundary has insufficient straight edge support')
        _, first, second = max(candidates, key = lambda item: item[0])
        direction = (second - first) / np.linalg.norm(second - first)
        normal = np.asarray([-direction[1], direction[0]], dtype = np.float32)
        lines.append((normal, np.float32(normal @ first)))
    corners = []
    for index in range(4):
        first, second = lines[index - 1], lines[index]
        matrix = np.stack((first[0], second[0]))
        if abs(float(np.linalg.det(matrix))) < .1:
            raise ValueError('target boundary lines are nearly parallel')
        corners.append(_single_solve(matrix, np.asarray([first[1], second[1]], dtype = np.float32)))
    return np.asarray(corners, dtype = np.float32)


def target_quad(rgb, markers, config):
    # Display-only edge detection. The resulting coordinates resample original FP32 RGB.
    height, width = rgb.shape[:2]
    scale = min(1., config['detection_size'] / max(height, width))
    small = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation = cv2.INTER_AREA)
    low, high = np.percentile(small, [1, 99])
    display = np.uint8(np.clip((small - low) / max(float(high - low), 1.), 0, 1) * 255 + .5)
    gray = cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
    blur = config['blur_size']
    if blur < 1 or blur % 2 != 1:
        raise ValueError('target blur_size must be positive and odd')
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, config['canny_low'], config['canny_high'])
    kernel = np.ones((config['close_size'], config['close_size']), dtype = np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    tolerance = config['edge_tolerance_px']
    support = cv2.dilate(edges, np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype = np.uint8))
    tag_centers = [np.asarray(value, dtype = np.float32).mean(0) * scale for value in markers.values()]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        fraction = area / (small.shape[0] * small.shape[1])
        if not config['min_area_fraction'] <= fraction <= config['max_area_fraction']:
            continue
        polygon = cv2.approxPolyDP(contour, config['polygon_epsilon'] * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = polygon[:, 0].astype(np.float32)
        if any(cv2.pointPolygonTest(polygon, tuple(map(float, center)), False) >= 0 for center in tag_centers):
            continue
        rectangle_area = np.prod(cv2.minAreaRect(contour)[1])
        if rectangle_area <= 0 or area / rectangle_area < config['min_rectangularity']:
            continue
        try:
            points = refine_quad(edges, points, config)
        except ValueError:
            continue
        if (points < 0).any() or (points[:, 0] >= small.shape[1]).any() or (points[:, 1] >= small.shape[0]).any():
            continue
        if not cv2.isContourConvex(points.reshape(-1, 1, 2)):
            continue
        if any(cv2.pointPolygonTest(points.reshape(-1, 1, 2), tuple(map(float, center)), False) >= 0 for center in tag_centers):
            continue
        edge_support = []
        for a, b in zip(points, np.roll(points, -1, axis = 0)):
            count = max(2, int(np.linalg.norm(b - a)))
            sample = np.rint(np.linspace(a, b, count)).astype(int)
            edge_support.append(float((support[sample[:, 1], sample[:, 0]] > 0).mean()))
        if min(edge_support) < config['min_edge_support']:
            continue
        candidates.append((area, points, edge_support))
    if not candidates:
        raise ValueError('no unambiguous four-sided target boundary; inspect boundary preview or capture again')
    _, points, edge_support = max(candidates, key = lambda item: item[0])
    # Order TL, TR, BR, BL; preserve the measured quadrilateral, not a bounding box.
    center = points.mean(0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    points = np.roll(points, -int(np.argmin(points.sum(1))), axis = 0) / np.float32(scale)
    distance = np.linalg.norm(points - points.mean(0), axis = 1, keepdims = True)
    inset = config['inset_pixels']
    if not 0 <= inset < float(distance.min()):
        raise ValueError('invalid target boundary inset')
    points += (points.mean(0) - points) * np.float32(inset) / distance
    return np.ascontiguousarray(points, dtype = np.float32), {
        'method': 'closed_edges_convex_quad', 'edge_support': edge_support,
        'detection_scale': scale, 'inset_source_pixels': inset, 'candidate_count': len(candidates)}


def detect_target_geometry(rgb, config):
    validate_target_config(config)
    source, record = marker_record(rgb, min_tags = config['registration']['min_shared_tags'])
    quad, evidence = target_quad(source, record['marker_corners_raw'], config['target'])
    size = config['warp_size']
    destination = np.asarray([[0, 0], [size, 0], [size, size], [0, size]], dtype = np.float32)
    matrix = _perspective_from_quad(quad, destination)
    record.update(homography = matrix.tolist(), quad = quad.tolist(), spatial_policy = 'robust_target',
                  target_detection = evidence)
    return matrix, record
