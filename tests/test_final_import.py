#!/usr/bin/env python3
"""最终测试本地子模块导入"""
import sys
import os
from pathlib import Path

print("测试本地子模块导入功能...")

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 测试OmniVoice本地导入
print("\n1. 测试OmniVoice本地导入...")
omni_path = project_root / "systems" / "OmniVoice"
if omni_path.exists():
    print(f"✅ OmniVoice目录存在: {omni_path}")
    
    # 尝试从本地模块导入
    try:
        # 将OmniVoice目录添加到sys.path
        sys.path.insert(0, str(omni_path))
        
        # 尝试导入omnivoice模块
        try:
            # 首先导入models.omnivoice
            from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
            print("✅ 成功从omnivoice.models.omnivoice导入OmniVoice和OmniVoiceConfig")
        except ImportError:
            # 回退到直接导入
            try:
                from omnivoice import OmniVoice, OmniVoiceConfig
                print("✅ 成功从omnivoice导入OmniVoice和OmniVoiceConfig")
            except ImportError as e:
                print(f"❌ 无法导入OmniVoice: {e}")
    except Exception as e:
        print(f"❌ 导入过程中出错: {e}")
else:
    print("❌ OmniVoice目录不存在")

# 测试LongCat-AudioDiT本地导入
print("\n2. 测试LongCat-AudioDiT本地导入...")
longcat_path = project_root / "systems" / "LongCat-AudioDiT"
if longcat_path.exists():
    print(f"✅ LongCat-AudioDiT目录存在: {longcat_path}")
    
    # 添加LongCat-AudioDiT到sys.path
    sys.path.insert(0, str(longcat_path))
    
    try:
        # 导入audiodit模块
        import audiodit
        print("✅ 成功导入audiodit模块")
        
        # 尝试导入AudioDiTModel
        try:
            from audiodit import AudioDiTModel
            print("✅ 成功从audiodit导入AudioDiTModel")
        except ImportError as e:
            print(f"❌ 无法导入AudioDiTModel: {e}")
    except Exception as e:
        print(f"❌ 导入audiodit失败: {e}")
else:
    print("❌ LongCat-AudioDiT目录不存在")

# 测试适配器创建
print("\n3. 测试适配器创建...")
try:
    from models.manager import TTSModelManager
    print("✅ 成功导入TTSModelManager")
    
    # 测试创建模型管理器（不加载模型）
    try:
        manager = TTSModelManager("k2-fsa/OmniVoice")
        print(f"✅ 成功创建TTSModelManager, framework: {manager.framework}")
        
        # 测试获取模型信息（不加载模型）
        try:
            info = manager.get_framework_info()
            print(f"✅ 成功获取模型信息: {info}")
        except Exception as e:
            print(f"⚠ 获取模型信息时出错: {e} (可能是预期的，因为模型未加载)")
    except Exception as e:
        print(f"❌ 创建TTSModelManager失败: {e}")
except ImportError as e:
    print(f"❌ 导入TTSModelManager失败: {e}")

# 测试FastAPI应用
print("\n4. 测试FastAPI应用...")
try:
    from api.endpoints import create_app
    from config import get_default_config
    
    print("✅ 成功导入create_app和get_default_config")
    
    # 创建配置
    config = get_default_config()
    print(f"✅ 成功获取默认配置: model={config.model_repo}, framework={config.framework}")
    
    # 创建应用
    app = create_app(config)
    print(f"✅ 成功创建FastAPI应用，路由数量: {len(app.routes)}")
    
    # 检查关键端点
    routes = {route.path: route.methods for route in app.routes if hasattr(route, 'path')}
    required_routes = ["/speakers", "/tts_to_audio/", "/health"]
    
    for route in required_routes:
        if route in routes:
            print(f"✅ 端点 {route} 存在")
        else:
            print(f"⚠ 端点 {route} 未找到")
    
except Exception as e:
    print(f"❌ FastAPI应用测试失败: {e}")

print("\n5. 测试说话人扫描...")
try:
    from utils.file_utils import scan_speakers
    
    # 检查samples目录
    samples_dir = project_root / "samples"
    if samples_dir.exists():
        print(f"✅ Samples目录存在: {samples_dir}")
        speakers = scan_speakers(samples_dir)
        print(f"✅ 找到 {len(speakers)} 个说话人: {[s['id'] for s in speakers]}")
    else:
        print("⚠ Samples目录不存在")
except Exception as e:
    print(f"❌ 说话人扫描测试失败: {e}")

print("\n" + "="*60)
print("本地导入测试完成")
print("="*60)