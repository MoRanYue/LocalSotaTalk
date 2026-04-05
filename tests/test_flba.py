#!/usr/bin/env python3
"""测试flba说话人检测"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.file_utils import scan_speakers, get_speaker_by_id

samples_dir = Path('samples')
print('扫描说话人...')
speakers = scan_speakers(samples_dir)
print(f'找到 {len(speakers)} 个说话人:')
for s in speakers:
    print(f"  ID: {s['id']}")
    print(f"    类型: {s['type']}")
    print(f"    音频路径: {s['audio_path']}")
    print(f"    文本路径: {s['text_path']}")
    print(f"    设计路径: {s['design_path']}")
    print(f"    设计描述: {s['design_description']}")
    print()

# 测试获取flba
print('测试获取flba:')
speaker = get_speaker_by_id(samples_dir, 'flba')
if speaker:
    print(f"✅ 找到flba说话人")
    print(f"   类型: {speaker['type']}")
    print(f"   音频路径: {speaker['audio_path']}")
    print(f"   设计描述: {speaker['design_description']}")
else:
    print('❌ 未找到flba说话人')
    
# 测试获取paimon
print('\n测试获取paimon:')
speaker = get_speaker_by_id(samples_dir, 'paimon')
if speaker:
    print(f"✅ 找到paimon说话人")
    print(f"   类型: {speaker['type']}")
    print(f"   音频路径: {speaker['audio_path']}")
else:
    print('❌ 未找到paimon说话人')