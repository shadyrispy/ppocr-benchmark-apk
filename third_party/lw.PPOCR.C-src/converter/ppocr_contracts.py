"""Machine-readable contracts for experimental PP-OCRv6 model assets.

These constants describe the exact Small and Medium validation assets currently
used by the analysis tools. They do not promote either variant to a released
model; they make every experimental conversion fail closed when the source
contract changes.
"""

from __future__ import annotations


PP_OCRV6_REC_WIDTHS = (192, 320, 480, 640, 960)

PP_OCRV6_SMALL_DET_SHA256 = "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e"
PP_OCRV6_SMALL_REC_SHA256 = "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634"
PP_OCRV6_SMALL_DICT_SHA256 = "118d0f0714ad2a37668c23d6541f2c3feb65b8214041265b567f7fd5b3365d8e"
PP_OCRV6_MEDIUM_REC_SHA256 = "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba"
PP_OCRV6_MEDIUM_DET_SHA256 = "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1"
PP_OCRV6_TINY_CLS_SHA256 = "dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2"
PP_OCRV6_SMALL_CLS_SHARED = True

PP_OCRV6_SMALL_REC_CLASSES = 18710
PP_OCRV6_SMALL_DICT_ENTRIES = 18708
PP_OCRV6_SMALL_REC_BASE_SHAPE = (1, 3, 48, 1)

SMALL_REC_DYNAMIC_OUTPUTS = (
    "Shape.1",
    "Shape.3",
    "Shape.7",
    "Shape.13",
    "Slice.1",
    "helper.slice.0",
)


def small_rec_metadata(width: int) -> dict[str, list[int]]:
    """Return the exact metadata output contract for one REC input width."""
    if width <= 0 or width % 8:
        raise ValueError("Small REC width must be a positive multiple of eight")
    steps = width // 8
    return {
        "Shape.1": [1, 120, 1, steps],
        "Shape.3": [1, 120, 1, steps],
        "Shape.7": [1, 8, steps, 15],
        "Shape.13": [1, 8, steps, 15],
        "Slice.1": [steps],
        "helper.slice.0": [1, 120],
    }
