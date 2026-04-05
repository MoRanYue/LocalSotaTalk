#!/usr/bin/env python3
"""最终服务器测试"""
import subprocess
import sys
import time
import requests
from pathlib import Path

def test_server():
    print("启动TTS服务器测试...")
    
    # 启动服务器
    cmd = [sys.executable, 'main.py', '--model', 'k2-fsa/OmniVoice', '--host', '127.0.0.1', '--port', '8003']
    print(f"命令: {' '.join(cmd)}")
    
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    print(f"服务器进程启动，PID: {server_process.pid}")
    print("等待服务器启动...")
    time.sleep(10)  # 等待更长时间确保服务器完全启动
    
    # 检查进程状态
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print(f"服务器已退出，退出码: {server_process.returncode}")
        print(f"stdout: {stdout[:500]}")
        print(f"stderr: {stderr[:500]}")
        return False
    else:
        print("服务器正在运行，开始测试...")
        
        all_tests_passed = True
        
        # 测试健康检查
        try:
            response = requests.get('http://127.0.0.1:8003/health', timeout=10)
            print(f"健康检查: {response.status_code}, {response.json()}")
            if response.status_code == 200:
                print("✅ 健康检查通过")
            else:
                print("❌ 健康检查失败")
                all_tests_passed = False
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            all_tests_passed = False
        
        # 测试CORS头部
        try:
            response = requests.get('http://127.0.0.1:8003/speakers', timeout=10)
            print(f"speakers端点: {response.status_code}")
            
            # 检查CORS头部
            cors_headers_present = False
            print("CORS头部检查:")
            for header, value in response.headers.items():
                if 'access-control' in header.lower():
                    print(f"  {header}: {value}")
                    cors_headers_present = True
            
            if cors_headers_present:
                print("✅ CORS头部正确")
            else:
                print("❌ CORS头部缺失")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ speakers测试失败: {e}")
            all_tests_passed = False
        
        # 测试TTS音频响应
        try:
            data = {
                'text': '你好，这是一个测试',
                'speaker_wav': 'paimon.wav',
                'language': 'zh-cn'
            }
            response = requests.post('http://127.0.0.1:8003/tts_to_audio/', json=data, timeout=60)  # 延长超时时间
            
            print(f"TTS响应状态: {response.status_code}")
            print(f"Content-Type: {response.headers.get('Content-Type')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'audio' in content_type:
                    print(f"✅ 正确的音频响应类型: {content_type}")
                    print(f"音频数据大小: {len(response.content)} 字节")
                    
                    # 验证是否为WAV文件
                    if response.content[:4] == b'RIFF':
                        print("✅ 有效的WAV文件 (RIFF头部)")
                    else:
                        print("⚠ 不是标准的WAV文件格式")
                        
                else:
                    print(f"❌ 错误的内容类型: {content_type}")
                    print(f"响应预览: {response.text[:200] if response.text else '无文本响应'}")
                    all_tests_passed = False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                print(f"错误详情: {response.text[:500]}")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ TTS测试失败: {e}")
            all_tests_passed = False
        
        # 停止服务器
        print("\n停止服务器...")
        server_process.terminate()
        server_process.wait()
        print("服务器已停止")
        
        return all_tests_passed

if __name__ == "__main__":
    print("=" * 60)
    print("TTS服务器最终测试")
    print("=" * 60)
    
    success = test_server()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)