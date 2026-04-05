#!/usr/bin/env python3
"""完整应用测试"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("完整应用测试")
print("="*60)

# 1. 测试所有模块导入
print("\n1. 测试所有模块导入...")

modules_to_test = [
    ("utils.constants", "SUPPORTED_LANGUAGES"),
    ("utils.file_utils", "scan_speakers"),
    ("models.base_adapter", "BaseTTSAdapter"),
    ("models.omnivoice_adapter", "OmniVoiceAdapter"),
    ("models.longcat_adapter", "LongCatAdapter"),
    ("models.manager", "TTSModelManager"),
    ("config", "get_default_config"),
    ("api.schemas", "TTSParams"),
    ("api.endpoints", "create_app"),
]

for module_name, attr_name in modules_to_test:
    try:
        exec(f"from {module_name} import {attr_name}")
        print(f"✅ {module_name}.{attr_name} 导入成功")
    except Exception as e:
        print(f"❌ {module_name}.{attr_name} 导入失败: {e}")

# 2. 测试配置
print("\n2. 测试配置...")
try:
    from config import get_default_config
    config = get_default_config()
    print(f"✅ 默认配置: model={config.model_repo}, framework={config.framework}")
    print(f"✅ Samples目录: {config.samples_dir}")
    print(f"✅ Output目录: {config.output_dir}")
except Exception as e:
    print(f"❌ 配置测试失败: {e}")

# 3. 测试模型管理器
print("\n3. 测试模型管理器...")
try:
    from models.manager import TTSModelManager
    
    # 测试OmniVoice检测
    manager1 = TTSModelManager("k2-fsa/OmniVoice")
    print(f"✅ OmniVoice管理器创建成功: framework={manager1.framework}")
    
    # 测试LongCat-AudioDiT检测
    manager2 = TTSModelManager("meituan-longcat/LongCat-AudioDiT-1B")
    print(f"✅ LongCat-AudioDiT管理器创建成功: framework={manager2.framework}")
    
    # 测试框架信息获取（不加载模型）
    try:
        info = manager1.get_framework_info()
        print(f"✅ 获取框架信息: {info}")
    except Exception as e:
        print(f"⚠ 获取框架信息失败 (可能模型未加载): {e}")
        
except Exception as e:
    print(f"❌ 模型管理器测试失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试FastAPI应用
print("\n4. 测试FastAPI应用...")
try:
    from api.endpoints import create_app
    from config import get_default_config
    
    config = get_default_config()
    app = create_app(config)
    
    # 检查路由
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append(route.path)
    
    print(f"✅ FastAPI应用创建成功，共 {len(routes)} 个路由")
    print(f"✅ 关键端点检查:")
    
    required_endpoints = [
        ("/", "根路径"),
        ("/health", "健康检查"),
        ("/speakers", "说话人列表"),
        ("/tts_to_audio", "TTS合成"),
    ]
    
    for path, desc in required_endpoints:
        if any(r.startswith(path) for r in routes):
            print(f"  ✅ {path} - {desc}")
        else:
            print(f"  ⚠ {path} - {desc} (未找到)")
    
except Exception as e:
    print(f"❌ FastAPI应用测试失败: {e}")

# 5. 测试本地子模块导入
print("\n5. 测试本地子模块导入...")
try:
    # 测试OmniVoice本地导入
    omni_path = project_root / "systems" / "OmniVoice"
    if omni_path.exists():
        sys.path.insert(0, str(omni_path))
        try:
            from omnivoice.models.omnivoice import OmniVoice
            print(f"✅ OmniVoice本地导入成功: {OmniVoice.__name__}")
        except ImportError as e:
            print(f"❌ OmniVoice本地导入失败: {e}")
    else:
        print("⚠ OmniVoice目录不存在")
    
    # 测试LongCat-AudioDiT本地导入
    longcat_path = project_root / "systems" / "LongCat-AudioDiT"
    if longcat_path.exists():
        sys.path.insert(0, str(longcat_path))
        try:
            import audiodit
            from audiodit import AudioDiTModel
            print(f"✅ LongCat-AudioDiT本地导入成功: {AudioDiTModel.__name__}")
        except ImportError as e:
            print(f"❌ LongCat-AudioDiT本地导入失败: {e}")
    else:
        print("⚠ LongCat-AudioDiT目录不存在")
        
except Exception as e:
    print(f"❌ 本地子模块导入测试失败: {e}")

# 6. 测试说话人扫描
print("\n6. 测试说话人扫描...")
try:
    from utils.file_utils import scan_speakers
    
    samples_dir = project_root / "samples"
    if samples_dir.exists():
        speakers = scan_speakers(samples_dir)
        print(f"✅ 扫描到 {len(speakers)} 个说话人")
        if speakers:
            for speaker in speakers[:3]:  # 显示前3个
                print(f"  - {speaker['id']}: {speaker['type']}")
    else:
        print("⚠ Samples目录不存在，创建示例目录...")
        os.makedirs(samples_dir, exist_ok=True)
        # 创建示例文件
        demo_wav = samples_dir / "demo.wav"
        demo_txt = samples_dir / "demo.txt"
        with open(demo_txt, 'w', encoding='utf-8') as f:
            f.write("这是一个示例音频。")
        print(f"✅ 创建示例目录和文件")
        
except Exception as e:
    print(f"❌ 说话人扫描测试失败: {e}")

print("\n" + "="*60)
print("完整应用测试完成")
print("="*60)