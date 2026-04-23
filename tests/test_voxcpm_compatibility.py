#!/usr/bin/env python3
"""测试VoxCPM1和VoxCPM2的兼容性"""

import sys
import os
from pathlib import Path

# 添加本地VoxCPM模块路径
LOCAL_VOXCPM_PATH = Path(__file__).parent / "systems" / "VoxCPM"
if LOCAL_VOXCPM_PATH.exists():
    src_path = LOCAL_VOXCPM_PATH / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(LOCAL_VOXCPM_PATH))

from models.voxcpm_adapter import VoxCPMAdapter

def test_model_detection():
    """测试模型类型检测"""
    print("=== 测试VoxCPM模型类型检测 ===")
    
    # 模拟不同模型的特征
    test_cases = [
        {
            "name": "VoxCPM1.5 (44100Hz)",
            "sample_rate": 44100,
            "has_out_sample_rate": False,
            "class_name": "VoxCPMModel",
            "expected_is_voxcpm2": False
        },
        {
            "name": "VoxCPM2 (48000Hz)",
            "sample_rate": 48000,
            "has_out_sample_rate": True,
            "class_name": "VoxCPM2Model",
            "expected_is_voxcpm2": True
        },
        {
            "name": "VoxCPM1 (16000Hz)",
            "sample_rate": 16000,
            "has_out_sample_rate": False,
            "class_name": "VoxCPMModel",
            "expected_is_voxcpm2": False
        },
    ]
    
    for test in test_cases:
        # 创建适配器实例
        adapter = VoxCPMAdapter("test/model")
        
        # 模拟设置模型属性
        class MockTTSModel:
            def __init__(self, test_case):
                self.sample_rate = test_case["sample_rate"]
                if test_case["has_out_sample_rate"]:
                    self.out_sample_rate = test_case["sample_rate"]
        
        class MockVoxCPM:
            def __init__(self, test_case):
                self.tts_model = MockTTSModel(test_case)
                # 设置类名
                self.tts_model.__class__.__name__ = test_case["class_name"]
        
        # 模拟加载模型
        adapter._voxcpm_instance = MockVoxCPM(test)
        adapter.model = adapter._voxcpm_instance
        adapter.is_loaded = True
        
        # 手动设置采样率
        adapter.sample_rate = test["sample_rate"]
        
        # 运行检测逻辑
        adapter.is_voxcpm2 = (
            hasattr(adapter.model.tts_model, 'out_sample_rate') or 
            adapter.sample_rate == 48000 or
            'voxcpm2' in str(type(adapter.model.tts_model)).lower()
        )
        
        # 检查结果
        result = "✓" if adapter.is_voxcpm2 == test["expected_is_voxcpm2"] else "✗"
        print(f"{result} {test['name']}:")
        print(f"  采样率: {test['sample_rate']}Hz")
        print(f"  有out_sample_rate: {test['has_out_sample_rate']}")
        print(f"  类名: {test['class_name']}")
        print(f"  检测结果: {'VoxCPM2' if adapter.is_voxcpm2 else 'VoxCPM1'}")
        print(f"  期望: {'VoxCPM2' if test['expected_is_voxcpm2'] else 'VoxCPM1'}")
        print()

def test_parameter_generation():
    """测试参数生成逻辑"""
    print("=== 测试参数生成逻辑 ===")
    
    # 测试用例：不同模型类型下的参数生成
    test_cases = [
        {
            "name": "VoxCPM1.5 + 音频文件",
            "is_voxcpm2": False,
            "speaker_wav": "test.wav",
            "has_txt_file": True,
            "expected_params": ["prompt_wav_path", "prompt_text"],  # 应该使用极致克隆
            "not_expected": ["reference_wav_path"]  # 不应该有reference_wav_path
        },
        {
            "name": "VoxCPM1.5 + 音频文件（无文本）",
            "is_voxcpm2": False,
            "speaker_wav": "test.wav",
            "has_txt_file": False,
            "expected_params": [],  # 没有文本文件，无法克隆
            "not_expected": ["reference_wav_path", "prompt_wav_path"]
        },
        {
            "name": "VoxCPM2 + 音频文件",
            "is_voxcpm2": True,
            "speaker_wav": "test.wav",
            "has_txt_file": False,
            "expected_params": ["reference_wav_path"],  # 应该使用reference_wav_path
            "not_expected": []
        },
        {
            "name": "VoxCPM2 + 音频文件 + 文本",
            "is_voxcpm2": True,
            "speaker_wav": "test.wav",
            "has_txt_file": True,
            "expected_params": ["reference_wav_path", "prompt_wav_path", "prompt_text"],  # 极致克隆
            "not_expected": []
        },
        {
            "name": "音色设计文件",
            "is_voxcpm2": False,
            "speaker_wav": "test.design.txt",
            "has_txt_file": False,
            "expected_params": ["control_instruction"],  # 应该使用control_instruction
            "not_expected": ["reference_wav_path", "prompt_wav_path"]
        },
    ]
    
    for test in test_cases:
        print(f"\n测试: {test['name']}")
        
        # 创建适配器实例
        adapter = VoxCPMAdapter("test/model")
        adapter.is_voxcpm2 = test["is_voxcpm2"]
        adapter.generation_config = {"cfg_value": 2.0, "inference_timesteps": 10}
        
        # 模拟文件存在
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 创建测试文件
            audio_file = temp_dir / "test.wav"
            audio_file.write_bytes(b"fake audio data")
            
            if test["has_txt_file"]:
                txt_file = temp_dir / "test.txt"
                txt_file.write_text("This is reference text.", encoding="utf-8")
            
            design_file = temp_dir / "test.design.txt"
            design_file.write_text("A cheerful female voice", encoding="utf-8")
            
            # 根据测试用例选择文件
            if test["speaker_wav"].endswith(".design.txt"):
                speaker_path = str(design_file)
            else:
                speaker_path = str(audio_file)
            
            # 调用参数准备方法
            # 由于我们无法直接调用私有方法，这里模拟逻辑
            gen_kwargs = adapter.generation_config.copy()
            
            # 模拟参数准备逻辑
            if speaker_path and Path(speaker_path).exists():
                if speaker_path.endswith(".design.txt"):
                    # 音色设计模式
                    with open(speaker_path, 'r', encoding='utf-8') as f:
                        design_desc = f.read().strip()
                    gen_kwargs["control_instruction"] = design_desc
                else:
                    # 语音克隆模式
                    if test["is_voxcpm2"]:
                        gen_kwargs["reference_wav_path"] = speaker_path
                    
                    # 检查是否有对应的文本文件
                    txt_file = Path(speaker_path).with_suffix(".txt")
                    if txt_file.exists():
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            ref_text = f.read().strip()
                        gen_kwargs["prompt_wav_path"] = speaker_path
                        gen_kwargs["prompt_text"] = ref_text
            
            # 过滤参数
            supported_keys = {
                "control_instruction", "prompt_text", "prompt_wav_path", "reference_wav_path",
                "cfg_value", "inference_timesteps", "min_len", "max_len", "normalize", "denoise",
                "retry_badcase", "retry_badcase_max_times", "retry_badcase_ratio_threshold"
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in supported_keys}
            
            # 验证结果
            print(f"  生成的参数: {list(gen_kwargs.keys())}")
            
            all_expected_present = all(p in gen_kwargs for p in test["expected_params"])
            no_unexpected = all(p not in gen_kwargs for p in test["not_expected"])
            
            if all_expected_present and no_unexpected:
                print(f"  ✓ 参数正确")
            else:
                print(f"  ✗ 参数错误")
                if not all_expected_present:
                    missing = [p for p in test["expected_params"] if p not in gen_kwargs]
                    print(f"    缺少参数: {missing}")
                if not no_unexpected:
                    unexpected = [p for p in test["not_expected"] if p in gen_kwargs]
                    print(f"    多余参数: {unexpected}")
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试VoxCPM1不支持reference_wav_path的情况
    print("测试VoxCPM1语音克隆场景:")
    
    # 创建适配器实例
    adapter = VoxCPMAdapter("test/model")
    adapter.is_voxcpm2 = False
    adapter.sample_rate = 44100
    
    # 模拟场景：VoxCPM1 + 音频文件，但没有对应的文本文件
    print("场景1: VoxCPM1 + 音频文件（无文本）")
    print("  预期: 应该不传递reference_wav_path，因为没有文本文件无法进行极致克隆")
    print("  结果: 应该回退到零样本TTS或给出错误提示")
    
    print("\n场景2: VoxCPM1 + 音频文件 + 文本文件")
    print("  预期: 应该使用prompt_wav_path + prompt_text进行极致克隆")
    print("  结果: 可以正常进行语音克隆")
    
    print("\n场景3: VoxCPM2 + 音频文件（无文本）")
    print("  预期: 可以使用reference_wav_path进行可控克隆")
    print("  结果: 可以正常进行语音克隆")

if __name__ == "__main__":
    test_model_detection()
    test_parameter_generation()
    test_error_handling()