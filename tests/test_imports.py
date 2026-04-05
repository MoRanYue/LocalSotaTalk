#!/usr/bin/env python3
"""测试必要的导入"""
import sys

print("测试TTS框架导入...")

# 测试OmniVoice
try:
    import omnivoice
    print("✅ OmniVoice 已安装")
except ImportError as e:
    print(f"❌ OmniVoice 未安装: {e}")
    print("安装命令: pip install omnivoice 或 pip install git+https://github.com/k2-fsa/OmniVoice.git")

# 测试torch
try:
    import torch
    print(f"✅ PyTorch 已安装: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch 未安装: {e}")

# 测试transformers
try:
    import transformers
    print(f"✅ Transformers 已安装: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers 未安装: {e}")

# 测试LongCat-AudioDiT依赖
try:
    import diffusers
    print(f"✅ Diffusers 已安装: {diffusers.__version__}")
except ImportError as e:
    print(f"⚠ Diffusers 未安装 (LongCat-AudioDiT需要): {e}")

print("\n测试本地模块导入...")

# 测试本地模块
try:
    from models.manager import TTSModelManager
    print("✅ TTSModelManager 可导入")
except ImportError as e:
    print(f"❌ TTSModelManager 导入失败: {e}")

try:
    from api.endpoints import create_app
    print("✅ API端点模块 可导入")
except ImportError as e:
    print(f"❌ API端点模块 导入失败: {e}")

try:
    from config import get_default_config
    print("✅ 配置模块 可导入")
except ImportError as e:
    print(f"❌ 配置模块 导入失败: {e}")

print("\n检查默认配置...")
try:
    config = get_default_config()
    print(f"默认模型: {config.model_repo}")
    print(f"框架类型: {config.framework}")
    print(f"样本目录: {config.samples_dir}")
    print(f"输出目录: {config.output_dir}")
except Exception as e:
    print(f"配置获取失败: {e}")