"""Stable public import surface for FlashVSR+ inference.

The implementation remains in the existing ``inference`` package so current
module and CLI entry points continue to work unchanged.
"""

from ..inference import (
    ColorCorrectionConfig,
    DecoderConfig,
    FlashVSRInferencePipeline,
    InferenceConfig,
    InferenceRequest,
    InferenceResult,
    ModelBundle,
    ModelPaths,
    StreamConfig,
    build_inference_pipeline,
    stream_config_for_mode,
)

__all__ = [
    "ColorCorrectionConfig",
    "DecoderConfig",
    "FlashVSRInferencePipeline",
    "InferenceConfig",
    "InferenceRequest",
    "InferenceResult",
    "ModelBundle",
    "ModelPaths",
    "StreamConfig",
    "build_inference_pipeline",
    "stream_config_for_mode",
]

__version__ = "0.1.0"
