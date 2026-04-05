#!/usr/bin/env python3
"""启动服务器测试"""
import subprocess
import time
import sys
from pathlib import Path

def start_server():
    """启动TTS服务器"""
    print("启动TTS服务器...")
    
    # 使用subprocess启动服务器
    cmd = [sys.executable, "main.py", "--model", "k2-fsa/OmniVoice", "--host", "127.0.0.1", "--port", "8001"]
    
    try:
        # 启动进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"服务器进程已启动 (PID: {process.pid})")
        
        # 等待几秒钟让服务器启动
        time.sleep(5)
        
        # 检查进程状态
        if process.poll() is not None:
            # 进程已退出
            stdout, stderr = process.communicate()
            print(f"服务器已退出，退出码: {process.returncode}")
            print(f"标准输出: {stdout}")
            print(f"标准错误: {stderr}")
            return False, None
        else:
            print("服务器正在运行...")
            return True, process
            
    except Exception as e:
        print(f"启动服务器失败: {e}")
        return False, None

def test_server():
    """测试服务器响应"""
    import requests
    
    print("\n测试服务器响应...")
    
    try:
        # 测试健康检查
        response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        print(f"健康检查状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        
        # 测试CORS头部
        print(f"\nCORS头部检查:")
        for header in response.headers:
            if 'access-control' in header.lower():
                print(f"  {header}: {response.headers[header]}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TTS服务器启动测试")
    print("=" * 60)
    
    # 启动服务器
    success, process = start_server()
    
    if success and process:
        try:
            # 测试服务器
            if test_server():
                print("\n✅ 服务器启动成功并通过测试")
            else:
                print("\n❌ 服务器测试失败")
                
            # 等待用户输入后停止服务器
            input("\n按Enter键停止服务器...")
            process.terminate()
            process.wait()
            print("服务器已停止")
            
        except KeyboardInterrupt:
            print("\n停止服务器...")
            if process:
                process.terminate()
                process.wait()
        except Exception as e:
            print(f"测试过程中出错: {e}")
            if process:
                process.terminate()
                process.wait()
    else:
        print("❌ 服务器启动失败")