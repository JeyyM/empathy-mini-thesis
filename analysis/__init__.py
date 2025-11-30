"""Analysis package for multimodal emotion data.

Provides reusable loader, normalizer and analysis functions to compare
facial/voice/fusion data across groups (Neutral/Opposing/Similar).
"""

from . import loader
from . import normalizer
from . import analysis as analysis_module

__all__ = ["loader", "normalizer", "analysis_module"]
