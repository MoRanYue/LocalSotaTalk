#!/usr/bin/env python3
"""测试OmniVoice模型加载"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("测试OmniVoice模型加载...")

try:
    # 首先测试本地导入
    omni_path = project_root / "systems" / "OmniVoice"
    if omni_path.exists():
        sys.path.insert(0, str(omni_path))
        print(f"✅ OmniVoice路径已添加: {omni_path}")
    else:
        print("❌ OmniVoice目录不存在")
        sys.exit(1)
    
    # 导入OmniVoice
    try:
        from omnivoice.models.omnivoice import OmniVoice
        print("✅ 成功从本地子模块导入OmniVoice")
    except ImportError as e:
        print(f"❌ 导入OmniVoice失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 尝试加载模型（不实际下载，会失败，但可以测试导入）
    print("\n测试模型加载逻辑...")
    
    # 创建适配器实例
    from models.omnivoice_adapter import OmniVoiceAdapter
    
    try:
        adapter = OmniVoiceAdapter("k2-fsa/OmniVoice")
        print(f"✅ 创建OmniVoiceAdapter成功")
        
        # 尝试加载模型（可能会因为网络下载而失败，但可以测试到accelerate错误之前）
        print("尝试加载模型（可能会因网络下载失败，但会测试accelerate依赖）...")
        try:
            adapter.load_model()
            print("✅ 模型加载成功！")
        except Exception as e:
            error_msg = str(e)
            print(f"模型加载错误: {error_msg}")
            
            # 检查是否是accelerate相关错误
            if "accelerate" in error_msg.lower():
                print("❌ 错误：仍然有accelerate依赖问题")
            elif "device_map" in error_msg.lower():
                print("❌ 错误：device_map相关问题")
            elif "no module named" in error_msg.lower():
                print("❌ 错误：缺少模块")
            elif "download" in error_msg.lower() or "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                print("⚠ 错误：网络下载问题（非代码问题）")
                print("   这可能是预期的，因为需要下载模型文件")
            else:
                print(f"⚠ 其他错误: {type(e).__name__}")
                
    except Exception as e:
        print(f"❌ 创建适配器失败: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")