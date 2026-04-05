#!/usr/bin/env python3
"""测试从本地子模块导入OmniVoice和LongCat-AudioDiT"""
import sys
import os
from pathlib import Path

print("测试本地子模块导入...")

# 添加OmniVoice路径
omni_path = Path(__file__).parent / "systems" / "OmniVoice"
longcat_path = Path(__file__).parent / "systems" / "LongCat-AudioDiT"

print(f"OmniVoice路径: {omni_path}")
print(f"LongCat-AudioDiT路径: {longcat_path}")

# 测试OmniVoice路径是否存在
if omni_path.exists():
    print("✅ OmniVoice目录存在")
    sys.path.insert(0, str(omni_path))
    
    # 列出omnivoice目录内容
    omni_module = omni_path / "omnivoice"
    if omni_module.exists():
        print(f"✅ omnivoice模块目录存在: {list(omni_module.iterdir())}")
        
        # 测试导入omnivoice
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("omnivoice", omni_module / "__init__.py")
            if spec is not None:
                omnivoice = importlib.util.module_from_spec(spec)
                sys.modules["omnivoice"] = omnivoice
                try:
                    spec.loader.exec_module(omnivoice)
                    print("✅ 成功加载omnivoice模块")
                    
                    # 测试导入OmniVoice类
                    try:
                        from omnivoice.models.omnivoice import OmniVoice
                        print("✅ 成功从omnivoice.models.omnivoice导入OmniVoice")
                    except ImportError as e:
                        print(f"❌ 无法从omnivoice.models.omnivoice导入: {e}")
                        # 尝试直接导入
                        try:
                            from omnivoice import OmniVoice
                            print("✅ 成功从omnivoice导入OmniVoice")
                        except ImportError as e2:
                            print(f"❌ 无法从omnivoice直接导入: {e2}")
                except Exception as e:
                    print(f"❌ 执行模块失败: {e}")
            else:
                print("❌ 无法创建模块规范")
        except Exception as e:
            print(f"❌ 导入测试失败: {e}")
    else:
        print("❌ omnivoice模块目录不存在")
else:
    print("❌ OmniVoice目录不存在")

print("\n" + "="*60)

# 测试LongCat-AudioDiT路径是否存在
if longcat_path.exists():
    print("✅ LongCat-AudioDiT目录存在")
    
    # 查看audiodit目录
    audiodit_module = longcat_path / "audiodit"
    if audiodit_module.exists():
        print(f"✅ audiodit模块目录存在: {list(audiodit_module.iterdir())}")
        
        # 添加LongCat-AudioDiT到sys.path
        sys.path.insert(0, str(longcat_path))
        
        # 测试导入audiodit
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("audiodit", audiodit_module / "__init__.py")
            if spec is not None:
                audiodit = importlib.util.module_from_spec(spec)
                sys.modules["audiodit"] = audiodit
                try:
                    spec.loader.exec_module(audiodit)
                    print("✅ 成功加载audiodit模块")
                    
                    # 测试导入AudioDiTModel
                    try:
                        from audiodit import AudioDiTModel
                        print("✅ 成功从audiodit导入AudioDiTModel")
                    except ImportError as e:
                        print(f"❌ 无法从audiodit导入AudioDiTModel: {e}")
                except Exception as e:
                    print(f"❌ 执行模块失败: {e}")
            else:
                print("❌ 无法创建模块规范")
        except Exception as e:
            print(f"❌ 导入测试失败: {e}")
    else:
        print("❌ audiodit模块目录不存在")
else:
    print("❌ LongCat-AudioDiT目录不存在")

print("\n" + "="*60)

# 测试从现有适配器导入
print("测试从现有适配器导入...")
try:
    from models.omnivoice_adapter import OmniVoiceAdapter
    print("✅ 成功导入OmniVoiceAdapter")
    
    # 测试创建适配器实例（不加载模型）
    try:
        adapter = OmniVoiceAdapter("k2-fsa/OmniVoice")
        print("✅ 成功创建OmniVoiceAdapter实例")
    except Exception as e:
        print(f"❌ 创建OmniVoiceAdapter实例失败: {e}")
except ImportError as e:
    print(f"❌ 导入OmniVoiceAdapter失败: {e}")

try:
    from models.longcat_adapter import LongCatAdapter
    print("✅ 成功导入LongCatAdapter")
    
    # 测试创建适配器实例（不加载模型）
    try:
        adapter = LongCatAdapter("meituan-longcat/LongCat-AudioDiT-1B")
        print("✅ 成功创建LongCatAdapter实例")
    except Exception as e:
        print(f"❌ 创建LongCatAdapter实例失败: {e}")
except ImportError as e:
    print(f"❌ 导入LongCatAdapter失败: {e}")

print("\n" + "="*60)
print("导入测试完成")