"""
hyperspeed/__init__.py
HyperSpeed: Ultra-fast CPU LLM inference
"""

__version__ = "0.1.0"

from .core import HyperSpeedEngine, convert_model

__all__ = ["HyperSpeedEngine", "convert_model"]
