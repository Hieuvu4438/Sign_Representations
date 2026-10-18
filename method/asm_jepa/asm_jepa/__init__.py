"""ASM-JEPA: anchored state-motion JEPA for isolated sign skeletons (proposal15).

The package root deliberately imports nothing heavy: pose tools (cv2, rtmlib, onnxruntime)
are imported lazily inside the extraction CLIs only.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "coco133_to_aslc114_v1"
CACHE_VERSION = "asm_pose_npz_v1"
