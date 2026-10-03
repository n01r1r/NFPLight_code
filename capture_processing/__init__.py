"""Float32 sensor-capture processing primitives.

The package deliberately keeps unpacking, demosaicing, geometry, photometry,
and model feature tracing separate.  Display images are never used as numeric
inputs.  ``demosaic_ahd`` is supplied by :mod:`capture_processing.demosaic`.
"""

from .geometry import area_resize, detect_marker_geometry, marker_rectify, rectify_stages, warp_perspective
from .photometry import CONDITIONS, apply_compensation, apply_linear_color, read_color_transform
from .raw import RawCapture, libraw_ahd_reference, unpack_dng
from .trace import assemble_feature_trace

__all__ = [
    "RawCapture",
    "unpack_dng",
    "libraw_ahd_reference",
    "warp_perspective",
    "area_resize",
    "marker_rectify",
    "rectify_stages",
    "detect_marker_geometry",
    "apply_compensation",
    "CONDITIONS",
    "read_color_transform",
    "apply_linear_color",
    "assemble_feature_trace",
]
