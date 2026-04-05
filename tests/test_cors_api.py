#!/usr/bin/env python3
"""测试CORS和API响应格式"""
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
import time
import soundfile as sf
import io

def test_cors():
    """测试CORS头部"""
    print("测试CORS头部...")
    
    # 检查/speakers端点的CORS头部
    try:
        response = requests.get("http://127.0.0.1:8000/speakers")
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        # 检查CORS相关头部
        cors_headers = [
            'Access-Control-Allow-Origin',
            'Access-Control-Allow-Methods',
            'Access-Control-Allow-Headers',
            'Access-Control-Allow-Credentials'
        ]
        
        for header in cors_headers:
            if header in response.headers:
                print(f"✅ {header}: {response.headers[header]}")
            else:
                print(f"⚠ {header}: 未找到")
                
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ CORS测试失败: {e}")
        return False

def test_tts_audio_response():
    """测试TTS音频响应格式"""
    print("\n测试TTS音频响应格式...")
    
    # 构建请求数据
    data = {
        "text": "你好，这是一个测试",
        "speaker_wav": "paimon.wav",
        "language": "zh-cn"
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/tts_to_audio/",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', '未找到')}")
        print(f"Content-Length: {response.headers.get('Content-Length', '未找到')}")
        print(f"其他头: {dict(response.headers)}")
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'audio' in content_type:
                print(f"✅ 正确的音频响应类型: {content_type}")
                
                # 尝试读取音频数据
                try:
                    audio_data = response.content
                    print(f"✅ 收到音频数据大小: {len(audio_data)} 字节")
                    
                    # 验证是否为有效的WAV文件
                    audio_buffer = io.BytesIO(audio_data)
                    try:
                        audio, sample_rate = sf.read(audio_buffer)
                        print(f"✅ 有效的WAV文件: {len(audio)} 样本点, {sample_rate} Hz")
                        return True
                    except Exception as e:
                        print(f"❌ 无效的音频数据: {e}")
                        return False
                        
                except Exception as e:
                    print(f"❌ 处理音频数据失败: {e}")
                    return False
            else:
                print(f"❌ 错误的响应类型: {content_type}")
                print(f"响应内容前100字节: {response.content[:100]}")
                return False
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            print(f"错误信息: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ TTS测试失败: {e}")
        return False

def test_speakers_endpoint():
    """测试speakers端点"""
    print("\n测试speakers端点...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/speakers")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ 有效的JSON响应")
                print(f"✅ 说话人数量: {len(data)}")
                for speaker in data[:3]:  # 只显示前3个
                    print(f"  - {speaker.get('name', '未知')} ({speaker.get('type', '未知')})")
                return True
            except Exception as e:
                print(f"❌ JSON解析失败: {e}")
                return False
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ speakers测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CORS和API响应测试")
    print("=" * 60)
    
    # 等待服务器启动（如果服务器已经运行）
    time.sleep(2)
    
    # 运行测试
    cors_ok = test_cors()
    speakers_ok = test_speakers_endpoint()
    tts_ok = test_tts_audio_response()
    
    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"CORS测试: {'✅ 通过' if cors_ok else '❌ 失败'}")
    print(f"Speakers端点: {'✅ 通过' if speakers_ok else '❌ 失败'}")
    print(f"TTS音频响应: {'✅ 通过' if tts_ok else '❌ 失败'}")
    print("=" * 60)