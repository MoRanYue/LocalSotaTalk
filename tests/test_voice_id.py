#!/usr/bin/env python3
"""测试voice_id功能"""
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

try:
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
        print()
    
    # 测试获取paimon
    print('测试获取paimon:')
    speaker = get_speaker_by_id(samples_dir, 'paimon')
    if speaker:
        print(f"✅ 找到paimon说话人")
        print(f"   音频路径: {speaker['audio_path']}")
        print(f"   完整路径: {Path(speaker['audio_path']).resolve()}")
        print(f"   文件存在: {Path(speaker['audio_path']).exists()}")
    else:
        print('❌ 未找到paimon说话人')
        
    # 测试获取不存在的说话人
    print('\n测试获取不存在的说话人:')
    speaker = get_speaker_by_id(samples_dir, 'nonexistent')
    if speaker:
        print(f"❌ 意外找到不存在的说话人")
    else:
        print(f"✅ 正确返回None")
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()