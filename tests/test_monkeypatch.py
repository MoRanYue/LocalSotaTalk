#!/usr/bin/env python3
"""测试猴子补丁修复accelerate问题"""
import sys
import os
from pathlib import Path
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 添加OmniVoice路径
omni_path = project_root / "systems" / "OmniVoice"
if omni_path.exists():
    sys.path.insert(0, str(omni_path))

print("测试猴子补丁修复...")

# 猴子补丁：拦截HiggsAudioV2TokenizerModel.from_pretrained
try:
    from transformers import AutoTokenizer, AutoFeatureExtractor
    import transformers
    
    # 保存原始方法
    original_from_pretrained = None
    
    # 尝试导入HiggsAudioV2TokenizerModel
    try:
        from transformers import HiggsAudioV2TokenizerModel
        original_from_pretrained = HiggsAudioV2TokenizerModel.from_pretrained
    except ImportError:
        print("⚠ HiggsAudioV2TokenizerModel不可用，可能在其他模块中")
    
    # 创建一个猴子补丁版本
    def patched_from_pretrained(pretrained_model_name_or_path, **kwargs):
        print(f"猴子补丁: HiggsAudioV2TokenizerModel.from_pretrained调用，移除device_map参数")
        # 移除device_map参数
        if 'device_map' in kwargs:
            print(f"  移除device_map参数: {kwargs['device_map']}")
            del kwargs['device_map']
        # 调用原始方法
        if original_from_pretrained:
            return original_from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            # 如果原始方法不可用，尝试使用基类方法
            from transformers import PreTrainedModel
            return PreTrainedModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
    
    # 应用猴子补丁
    if original_from_pretrained:
        HiggsAudioV2TokenizerModel.from_pretrained = patched_from_pretrained
        print("✅ 已应用HiggsAudioV2TokenizerModel猴子补丁")
    
    # 也尝试补丁PreTrainedModel.from_pretrained以处理device_map
    original_premodel_from_pretrained = transformers.PreTrainedModel.from_pretrained
    
    def patched_premodel_from_pretrained(pretrained_model_name_or_path, **kwargs):
        print(f"猴子补丁: PreTrainedModel.from_pretrained调用，处理device_map参数")
        # 如果device_map不是None，尝试设置为None
        if 'device_map' in kwargs and kwargs['device_map'] is not None:
            print(f"  修改device_map参数: {kwargs['device_map']} -> None")
            kwargs['device_map'] = None
        # 调用原始方法
        return original_premodel_from_pretrained(pretrained_model_name_or_path, **kwargs)
    
    transformers.PreTrainedModel.from_pretrained = patched_premodel_from_pretrained
    print("✅ 已应用PreTrainedModel猴子补丁")
    
except Exception as e:
    print(f"❌ 猴子补丁失败: {e}")
    import traceback
    traceback.print_exc()

# 现在测试加载模型
print("\n测试模型加载...")
try:
    from models.omnivoice_adapter import OmniVoiceAdapter
    
    adapter = OmniVoiceAdapter("k2-fsa/OmniVoice")
    print("✅ 创建适配器成功")
    
    try:
        adapter.load_model()
        print("✅ 模型加载成功！")
    except Exception as e:
        error_msg = str(e)
        print(f"模型加载错误: {error_msg}")
        if "accelerate" in error_msg.lower():
            print("❌ 仍然有accelerate依赖问题")
        else:
            print(f"其他错误: {type(e).__name__}")
            
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")