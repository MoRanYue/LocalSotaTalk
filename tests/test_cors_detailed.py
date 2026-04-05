#!/usr/bin/env python3
"""详细CORS测试"""
import subprocess
import sys
import time
import requests
from pathlib import Path

def print_headers(title, response):
    print(f"\n{title}:")
    for header, value in response.headers.items():
        print(f"  {header}: {value}")

def test_cors_detailed():
    print("启动服务器进行详细CORS测试...")
    
    # 启动服务器
    cmd = [sys.executable, 'main.py', '--model', 'k2-fsa/OmniVoice', '--host', '127.0.0.1', '--port', '8004']
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    print(f"服务器PID: {server_process.pid}")
    time.sleep(10)  # 等待服务器启动
    
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print(f"服务器已退出: {stderr[:500]}")
        return False
    
    try:
        # 测试1: 简单GET请求到/speakers
        print("\n" + "="*60)
        print("测试1: GET /speakers")
        response = requests.get('http://127.0.0.1:8004/speakers', timeout=10)
        print(f"状态码: {response.status_code}")
        print_headers("响应头", response)
        
        # 测试2: 带Origin头的GET请求
        print("\n" + "="*60)
        print("测试2: GET /speakers 带Origin头")
        headers = {'Origin': 'https://st.caughtwind.top:8000'}
        response = requests.get('http://127.0.0.1:8004/speakers', headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        print_headers("响应头", response)
        
        # 测试3: OPTIONS预检请求
        print("\n" + "="*60)
        print("测试3: OPTIONS /speakers 预检请求")
        headers = {
            'Origin': 'https://st.caughtwind.top:8000',
            'Access-Control-Request-Method': 'GET',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        response = requests.options('http://127.0.0.1:8004/speakers', headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        print_headers("响应头", response)
        
        # 测试4: POST请求到TTS端点
        print("\n" + "="*60)
        print("测试4: POST /tts_to_audio/ 带Origin头")
        headers = {'Origin': 'https://st.caughtwind.top:8000', 'Content-Type': 'application/json'}
        data = {
            'text': 'CORS测试',
            'speaker_wav': 'paimon.wav',
            'language': 'zh-cn'
        }
        response = requests.post('http://127.0.0.1:8004/tts_to_audio/', json=data, headers=headers, timeout=30)
        print(f"状态码: {response.status_code}")
        print_headers("响应头", response)
        
        # 测试5: OPTIONS预检请求到TTS端点
        print("\n" + "="*60)
        print("测试5: OPTIONS /tts_to_audio/ 预检请求")
        headers = {
            'Origin': 'https://st.caughtwind.top:8000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        response = requests.options('http://127.0.0.1:8004/tts_to_audio/', headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        print_headers("响应头", response)
        
        # 检查CORS关键头部
        print("\n" + "="*60)
        print("CORS关键头部检查:")
        test_urls = [
            ('http://127.0.0.1:8004/speakers', 'GET'),
            ('http://127.0.0.1:8004/tts_to_audio/', 'OPTIONS')
        ]
        
        for url, method in test_urls:
            if method == 'OPTIONS':
                headers = {
                    'Origin': 'https://st.caughtwind.top:8000',
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'Content-Type'
                }
                response = requests.options(url, headers=headers, timeout=10)
            else:
                headers = {'Origin': 'https://st.caughtwind.top:8000'}
                response = requests.get(url, headers=headers, timeout=10)
            
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers',
                'Access-Control-Allow-Credentials',
                'Access-Control-Max-Age'
            ]
            
            print(f"\n{method} {url}:")
            for header in cors_headers:
                value = response.headers.get(header, '未设置')
                print(f"  {header}: {value}")
        
        return True
        
    finally:
        # 停止服务器
        print("\n停止服务器...")
        server_process.terminate()
        server_process.wait()
        print("测试完成")

if __name__ == "__main__":
    print("详细CORS测试")
    print("="*60)
    test_cors_detailed()