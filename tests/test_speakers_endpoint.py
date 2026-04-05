#!/usr/bin/env python3
"""测试/speakers端点是否正确返回数组"""
import sys
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config
from api.endpoints import create_app
from fastapi.testclient import TestClient

def test_speakers_endpoint():
    """测试/speakers端点返回数组格式"""
    print("测试/speakers端点响应格式...")
    
    config = get_default_config()
    app = create_app(config)
    
    try:
        client = TestClient(app, base_url="http://testserver")
    except TypeError:
        client = TestClient(app)
    
    # 测试/speakers端点
    response = client.get("/speakers")
    assert response.status_code == 200, f"HTTP状态码错误: {response.status_code}"
    
    data = response.json()
    print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    # 验证响应是数组
    assert isinstance(data, list), f"响应不是数组，而是: {type(data)}"
    
    # 如果数组不为空，验证对象结构
    if len(data) > 0:
        first_speaker = data[0]
        print(f"第一个说话人: {first_speaker}")
        
        # 验证必要字段
        required_fields = ["name", "file_path", "voice_id"]
        for field in required_fields:
            assert field in first_speaker, f"缺少字段: {field}"
        
        # 验证voice_id等于name
        if first_speaker.get("voice_id") is not None:
            assert first_speaker["voice_id"] == first_speaker["name"], \
                f"voice_id({first_speaker['voice_id']})不等于name({first_speaker['name']})"
    
    print(f"✓ /speakers端点返回了数组，包含{len(data)}个说话人")
    return True

def test_speakers_list_endpoint():
    """测试/speakers_list端点返回包装对象格式"""
    print("\n测试/speakers_list端点响应格式...")
    
    config = get_default_config()
    app = create_app(config)
    
    try:
        client = TestClient(app, base_url="http://testserver")
    except TypeError:
        client = TestClient(app)
    
    # 测试/speakers_list端点
    response = client.get("/speakers_list")
    assert response.status_code == 200, f"HTTP状态码错误: {response.status_code}"
    
    data = response.json()
    print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    # 验证响应是对象
    assert isinstance(data, dict), f"响应不是对象，而是: {type(data)}"
    
    # 验证包含speakers和count字段
    assert "speakers" in data, "缺少speakers字段"
    assert "count" in data, "缺少count字段"
    
    # 验证speakers是数组
    assert isinstance(data["speakers"], list), "speakers字段不是数组"
    
    # 验证count与speakers长度一致
    assert data["count"] == len(data["speakers"]), \
        f"count({data['count']})与speakers长度({len(data['speakers'])})不一致"
    
    print(f"✓ /speakers_list端点返回了对象，包含{data['count']}个说话人")
    return True

def main():
    """运行所有端点测试"""
    print("=" * 60)
    print("测试TTS后端API端点格式")
    print("=" * 60)
    
    tests = [
        test_speakers_endpoint,
        test_speakers_list_endpoint,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
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
        print("✅ 所有端点格式测试通过!")
        print("\n端点格式总结:")
        print("- /speakers: 返回SpeakerInfo对象数组")
        print("- /speakers_list: 返回{speakers: [...], count: N}对象")
        return 0
    else:
        print(f"❌ {failed}个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())