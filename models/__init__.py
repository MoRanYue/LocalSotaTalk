"""TTS模型适配器模块"""
from .manager import TTSModelManager
from .base_adapter import BaseTTSAdapter
from .omnivoice_adapter import OmniVoiceAdapter
from .longcat_adapter import LongCatAdapter
from .voxcpm_adapter import VoxCPMAdapter

__all__ = [
    "TTSModelManager",
    "BaseTTSAdapter",
    "OmniVoiceAdapter",
    "LongCatAdapter",
    "VoxCPMAdapter"
]
