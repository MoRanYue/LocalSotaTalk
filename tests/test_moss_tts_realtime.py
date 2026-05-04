"""测试 MOSS-TTS-Realtime 适配器加载和合成"""
import sys
from pathlib import Path

# Add project root and local module paths
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
_LOCAL_MOSS_TTS_PATH = _PROJECT_ROOT / "systems" / "MOSS-TTS"
sys.path.insert(0, str(_LOCAL_MOSS_TTS_PATH))
_REALTIME_PATH = _LOCAL_MOSS_TTS_PATH / "moss_tts_realtime"
sys.path.insert(0, str(_REALTIME_PATH))

from models.moss_tts_adapter import MossTTSAdapter
import numpy as np

# 1. 测试适配器加载
print("=" * 60)
print("Test 1: Loading MossTTSAdapter...")
adapter = MossTTSAdapter("OpenMOSS-Team/MOSS-TTS-Realtime")
adapter.load_model()

info = adapter.get_model_info()
print(f"Framework: {info['framework']}")
print(f"Sample rate: {info['sample_rate']}")
print(f"Loaded: {adapter.is_loaded}")
print("Test 1 PASSED!")

# 2. 获取语言支持
print("=" * 60)
print("Test 2: Language support...")
langs = adapter.get_supported_languages()
print(f"Languages: {list(langs.keys())}")
print("Test 2 PASSED!")

# 3. 测试 TTS 合成
print("=" * 60)
print("Test 3: TTS synthesis...")
audio = adapter.synthesize(
    "Hello world, this is a test of the TTS system.",
    speaker_wav=None,
    language="en",
    max_new_tokens=512,
)
print(f"Audio shape: {audio.shape}")
print(f"Audio dtype: {audio.dtype}")
print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
assert len(audio.shape) == 1, "Audio should be mono (1D)"
assert audio.size > 0, "Audio should not be empty"
print("Test 3 PASSED!")

print("=" * 60)
print("ALL TESTS PASSED!")