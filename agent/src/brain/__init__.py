"""The brain behind the voice: delegation to other models and the local executor."""

from .bridge import BrainTask, Bridge
from .config import BrainConfig, BrainModel, default_brain, parse_brain

__all__ = ["BrainConfig", "BrainModel", "BrainTask", "Bridge", "default_brain", "parse_brain"]
