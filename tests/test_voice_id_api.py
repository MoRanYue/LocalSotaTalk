#!/usr/bin/env python3
"""测试voice_id API功能"""
import subprocess
import sys
import time
import requests
import json
from pathlib import Path

def run_voice_id_test():
    print("启动服务器测试voice_id API...")
    
    # 启动服务器
    cmd = [sys.executable, 'main.py', '--model', 'k2-fsa/OmniVoice', '--host', '127.0.0.1', '--port', '8005']
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
    time.sleep(12)  # 等待服务器启动和模型加载
    
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print(f"服务器已退出: {stderr[:500]}")
        return False
    
    try:
        print("服务器正在运行，开始测试...")
        all_tests_passed = True
        
        # 测试1: 获取说话人列表
        print("\n" + "="*60)
        print("测试1: 获取说话人列表")
        try:
            response = requests.get('http://127.0.0.1:8005/speakers', timeout=10)
            print(f"状态码: {response.status_code}")
            speakers = response.json()
            print(f"找到 {len(speakers)} 个说话人:")
            for speaker in speakers:
                print(f"  - {speaker['name']} (voice_id: {speaker['voice_id']})")
            
            # 检查是否包含paimon
            paimon_found = any(s['voice_id'] == 'paimon' for s in speakers)
            if paimon_found:
                print("✅ 找到paimon说话人")
            else:
                print("❌ 未找到paimon说话人")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ 获取说话人列表失败: {e}")
            all_tests_passed = False
        
        # 测试2: 使用voice_id进行TTS合成
        print("\n" + "="*60)
        print("测试2: 使用voice_id进行TTS合成")
        try:
            data = {
                'text': '你好，这是voice_id测试',
                'speaker_wav': 'paimon',  # 使用voice_id而不是路径
                'language': 'zh-cn'
            }
            
            print(f"发送TTS请求: {json.dumps(data, ensure_ascii=False)}")
            response = requests.post('http://127.0.0.1:8005/tts_to_audio/', 
                                   json=data, 
                                   timeout=60)
            
            print(f"状态码: {response.status_code}")
            print(f"Content-Type: {response.headers.get('Content-Type')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'audio' in content_type:
                    print(f"✅ TTS合成成功: {content_type}")
                    print(f"音频数据大小: {len(response.content)} 字节")
                    
                    # 验证是否为WAV文件
                    if response.content[:4] == b'RIFF':
                        print("✅ 有效的WAV文件 (RIFF头部)")
                    else:
                        print("⚠ 不是标准的WAV文件格式")
                        
                    # 保存测试文件
                    test_file = Path('output') / 'test_voice_id.wav'
                    test_file.parent.mkdir(exist_ok=True)
                    test_file.write_bytes(response.content)
                    print(f"音频已保存到: {test_file}")
                    
                else:
                    print(f"❌ 错误的内容类型: {content_type}")
                    print(f"响应预览: {response.text[:200]}")
                    all_tests_passed = False
            else:
                print(f"❌ TTS请求失败: {response.status_code}")
                print(f"错误详情: {response.text[:500]}")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ TTS合成测试失败: {e}")
            all_tests_passed = False
        
        # 测试3: 测试无效的voice_id
        print("\n" + "="*60)
        print("测试3: 测试无效的voice_id")
        try:
            data = {
                'text': '测试无效voice_id',
                'speaker_wav': 'nonexistent_voice_id',
                'language': 'zh-cn'
            }
            
            response = requests.post('http://127.0.0.1:8005/tts_to_audio/', 
                                   json=data, 
                                   timeout=10)
            
            print(f"状态码: {response.status_code}")
            if response.status_code == 404:
                print(f"✅ 正确返回404错误")
                print(f"错误消息: {response.json()['detail']}")
            else:
                print(f"❌ 期望404但得到 {response.status_code}")
                print(f"响应: {response.text[:200]}")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ 无效voice_id测试失败: {e}")
            all_tests_passed = False
        
        # 测试4: 使用voice_id进行文件保存
        print("\n" + "="*60)
        print("测试4: 使用voice_id保存到文件")
        try:
            data = {
                'text': '这是保存到文件的测试',
                'speaker_wav': 'paimon',
                'language': 'zh-cn',
                'file_name_or_path': 'voice_id_test_output.wav'
            }
            
            response = requests.post('http://127.0.0.1:8005/tts_to_file', 
                                   json=data, 
                                   timeout=60)
            
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 文件保存成功")
                print(f"文件路径: {result['file_path']}")
                print(f"时长: {result['duration']}秒")
                print(f"采样率: {result['sample_rate']}")
                
                # 检查文件是否存在
                file_path = Path(result['file_path'])
                if file_path.exists():
                    print(f"✅ 文件已创建: {file_path}")
                    file_size = file_path.stat().st_size
                    print(f"文件大小: {file_size} 字节")
                else:
                    print(f"❌ 文件未找到: {file_path}")
                    all_tests_passed = False
                    
            else:
                print(f"❌ 文件保存失败: {response.status_code}")
                print(f"错误详情: {response.text[:500]}")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ 文件保存测试失败: {e}")
            all_tests_passed = False
        
        return all_tests_passed
        
    finally:
        # 停止服务器
        print("\n停止服务器...")
        server_process.terminate()
        server_process.wait()
        print("测试完成")

if __name__ == "__main__":
    print("="*60)
    print("Voice_id API功能测试")
    print("="*60)
    
    success = run_voice_id_test()
    
    print("\n" + "="*60)
    if success:
        print("✅ 所有voice_id API测试通过!")
    else:
        print("❌ 部分测试失败")
    print("="*60)