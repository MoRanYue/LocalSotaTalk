#!/usr/bin/env python3
"""直接测试/speakers端点实现"""
import sys
import asyncio
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config
from api.endpoints import TTSAPI

async def test_speakers_direct():
    """直接测试get_speakers方法"""
    print("直接测试/speakers端点实现...")
    
    config = get_default_config()
    api = TTSAPI(config)
    
    # 直接调用get_speakers方法
    result = await api.get_speakers()
    
    # 验证结果是列表
    assert isinstance(result, list), f"返回值不是列表，而是: {type(result)}"
    
    print(f"✓ get_speakers()返回了列表，包含{len(result)}个说话人")
    
    # 如果列表不为空，验证第一个元素的字段
    if len(result) > 0:
        first_speaker = result[0]
        print(f"第一个说话人: {first_speaker}")
        
        # 验证必要字段
        required_fields = ["name", "file_path", "voice_id"]
        for field in required_fields:
            assert hasattr(first_speaker, field), f"缺少字段: {field}"
            print(f"  - {field}: {getattr(first_speaker, field)}")
        
        # 验证voice_id等于name
        if first_speaker.voice_id is not None:
            assert first_speaker.voice_id == first_speaker.name, \
                f"voice_id({first_speaker.voice_id})不等于name({first_speaker.name})"
            print(f"  ✓ voice_id等于name")
    
    return True

async def test_speakers_list_direct():
    """直接测试get_speakers_list方法"""
    print("\n直接测试/speakers_list端点实现...")
    
    config = get_default_config()
    api = TTSAPI(config)
    
    # 直接调用get_speakers_list方法
    result = await api.get_speakers_list()
    
    # 验证结果是字典
    assert isinstance(result, dict), f"返回值不是字典，而是: {type(result)}"
    
    # 验证包含speakers和count字段
    assert "speakers" in result, "缺少speakers字段"
    assert "count" in result, "缺少count字段"
    
    # 验证speakers是列表
    assert isinstance(result["speakers"], list), "speakers字段不是列表"
    
    # 验证count与speakers长度一致
    assert result["count"] == len(result["speakers"]), \
        f"count({result['count']})与speakers长度({len(result['speakers'])})不一致"
    
    print(f"✓ get_speakers_list()返回了字典，包含{result['count']}个说话人")
    return True

async def main():
    """运行所有直接测试"""
    print("=" * 60)
    print("直接测试TTS后端API端点实现")
    print("=" * 60)
    
    tests = [
        test_speakers_direct,
        test_speakers_list_direct,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__}失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"测试结果: {passed}通过, {failed}失败")
    print("=" * 60)
    
    if failed == 0:
        print("✅ 所有直接测试通过!")
        print("\n端点实现总结:")
        print("- /speakers端点: 返回SpeakerInfo对象数组")
        print("- /speakers_list端点: 返回{speakers: [...], count: N}对象")
        print("\n✅ API端点格式完全正确!")
        return 0
    else:
        print(f"❌ {failed}个测试失败")
        return 1

if __name__ == "__main__":
    # 运行异步测试
    exit_code = asyncio.run(main())
    sys.exit(exit_code)