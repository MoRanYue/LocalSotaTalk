#!/usr/bin/env python3
"""测试flba说话人的音频设计功能"""
import sys
import json
import subprocess
import time
import requests
from pathlib import Path

def test_flba_design():
    print("测试flba说话人的音频设计功能...")
    
    # 启动服务器
    cmd = [sys.executable, 'main.py', '--model', 'k2-fsa/OmniVoice', '--host', '127.0.0.1', '--port', '8006']
    print(f"启动服务器: {' '.join(cmd)}")
    
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    print(f"服务器PID: {server_process.pid}")
    print("等待服务器启动和模型加载...")
    time.sleep(15)  # 等待更长时间，因为需要加载模型
    
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print(f"服务器提前退出: {stderr[:500]}")
        return False
    
    try:
        print("\n=== 开始测试 ===")
        all_tests_passed = True
        
        # 测试1: 获取说话人列表，确认flba存在
        print("\n1. 测试说话人列表")
        try:
            response = requests.get('http://127.0.0.1:8006/speakers', timeout=10)
            print(f"状态码: {response.status_code}")
            speakers = response.json()
            print(f"找到 {len(speakers)} 个说话人:")
            
            flba_found = False
            for speaker in speakers:
                print(f"  - {speaker['name']} (voice_id: {speaker['voice_id']}, type: {speaker['type']})")
                if speaker['voice_id'] == 'flba':
                    flba_found = True
                    print(f"   设计描述: {speaker['design_description']}")
            
            if flba_found:
                print("✅ 找到flba说话人")
            else:
                print("❌ 未找到flba说话人")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ 获取说话人列表失败: {e}")
            all_tests_passed = False
        
        # 测试2: 使用flba进行音频设计合成
        print("\n2. 测试flba音频设计合成")
        try:
            data = {
                'text': 'Hello, this is a test of voice design functionality',
                'speaker_wav': 'flba',  # 使用flba的voice_id
                'language': 'en'
            }
            
            print(f"发送TTS请求: {json.dumps(data, ensure_ascii=False)}")
            response = requests.post('http://127.0.0.1:8006/tts_to_audio/', 
                                   json=data, 
                                   timeout=60)
            
            print(f"状态码: {response.status_code}")
            print(f"Content-Type: {response.headers.get('Content-Type')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'audio' in content_type:
                    print(f"✅ 音频设计合成成功: {content_type}")
                    print(f"音频数据大小: {len(response.content)} 字节")
                    
                    # 验证是否为WAV文件
                    if response.content[:4] == b'RIFF':
                        print("✅ 有效的WAV文件 (RIFF头部)")
                    else:
                        print("⚠ 不是标准的WAV文件格式")
                        
                    # 保存测试文件
                    test_file = Path('output') / 'flba_design_test.wav'
                    test_file.parent.mkdir(exist_ok=True)
                    test_file.write_bytes(response.content)
                    print(f"音频已保存到: {test_file}")
                    
                else:
                    print(f"❌ 错误的内容类型: {content_type}")
                    print(f"响应预览: {response.text[:200]}")
                    all_tests_passed = False
            elif response.status_code == 400:
                # 检查是否是模型不支持音频设计的错误
                error_detail = response.json().get('detail', '')
                print(f"❌ 请求失败: {error_detail}")
                if "does not support voice design" in error_detail:
                    print("⚠ 当前模型不支持音频设计功能")
                    print("  请切换到OmniVoice模型进行测试")
                all_tests_passed = False
            else:
                print(f"❌ TTS请求失败: {response.status_code}")
                print(f"错误详情: {response.text[:500]}")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ 音频设计合成测试失败: {e}")
            all_tests_passed = False
        
        # 测试3: 使用LongCat模型测试flba (应该不支持)
        print("\n3. 测试LongCat模型是否支持音频设计")
        try:
            # 先停止当前服务器
            server_process.terminate()
            server_process.wait()
            time.sleep(5)
            
            # 启动LongCat服务器
            cmd = [sys.executable, 'main.py', '--model', 'meituan-longcat/LongCat-AudioDiT-1B', '--host', '127.0.0.1', '--port', '8007']
            print(f"启动LongCat服务器: {' '.join(cmd)}")
            
            server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print("等待LongCat服务器启动...")
            time.sleep(20)  # LongCat可能需要更长时间
            
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                print(f"LongCat服务器提前退出: {stderr[:500]}")
                return all_tests_passed
            
            # 测试flba合成
            data = {
                'text': 'This should fail with LongCat',
                'speaker_wav': 'flba',
                'language': 'en'
            }
            
            response = requests.post('http://127.0.0.1:8007/tts_to_audio/', 
                                   json=data, 
                                   timeout=60)
            
            print(f"状态码: {response.status_code}")
            if response.status_code == 400 and "does not support voice design" in response.json().get('detail', ''):
                print("✅ LongCat正确返回不支持音频设计的错误")
            elif response.status_code == 404 and "not found" in response.json().get('detail', ''):
                print("✅ LongCat正确报告flba没有音频文件")
            else:
                print(f"❌ 期望的错误但得到 {response.status_code}")
                print(f"响应: {response.text[:200]}")
                all_tests_passed = False
                
        except Exception as e:
            print(f"⚠ LongCat测试出现异常: {e}")
            # 不因为这个失败，因为LongCat可能根本加载不了
        
        return all_tests_passed
        
    finally:
        # 停止服务器
        print("\n停止服务器...")
        if server_process.poll() is None:
            server_process.terminate()
            server_process.wait()
        print("测试完成")

if __name__ == "__main__":
    print("="*60)
    print("FLBA音频设计功能测试")
    print("="*60)
    
    success = test_flba_design()
    
    print("\n" + "="*60)
    if success:
        print("✅ FLBA音频设计测试通过!")
    else:
        print("❌ FLBA音频设计测试失败")
    print("="*60)