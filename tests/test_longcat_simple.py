#!/usr/bin/env python3
"""测试简化版LongCat适配器"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_adapter_structure():
    """测试适配器结构"""
    print("=== 测试LongCat适配器结构 ===")
    
    try:
        from models.longcat_adapter import LongCatAdapter
        
        # 创建适配器实例
        adapter = LongCatAdapter("meituan-longcat/LongCat-AudioDiT-1B")
        
        print("1. 适配器创建成功")
        print(f"   模型仓库: {adapter.model_repo}")
        print(f"   采样率: {adapter.sample_rate}")
        print(f"   生成配置: {adapter.generation_config}")
        
        # 测试方法存在性
        required_methods = [
            'load_model',
            'synthesize',
            '_prepare_generation_kwargs',
            '_estimate_duration',
            'get_supported_languages',
            'get_tts_settings',
            'update_tts_settings',
            'get_model_info',
            'synthesize_instructively'
        ]
        
        print("\n2. 检查必需方法:")
        for method in required_methods:
            if hasattr(adapter, method):
                print(f"   ✓ {method}")
            else:
                print(f"   ✗ {method} - 缺失!")
        
        # 测试配置
        print("\n3. 测试配置方法:")
        settings = adapter.get_tts_settings()
        print(f"   TTS设置: {settings}")
        
        languages = adapter.get_supported_languages()
        print(f"   支持语言: {languages}")
        
        model_info = adapter.get_model_info()
        print(f"   模型信息: {model_info}")
        
        print("\n✅ 适配器结构测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_kwargs():
    """测试生成参数准备"""
    print("\n=== 测试生成参数准备 ===")
    
    try:
        from models.longcat_adapter import LongCatAdapter
        
        # 创建适配器实例（不加载模型）
        adapter = LongCatAdapter("meituan-longcat/LongCat-AudioDiT-1B")
        
        # 模拟加载状态
        adapter.is_loaded = True
        adapter.model = type('MockModel', (), {'device': 'cpu'})()
        adapter.tokenizer = type('MockTokenizer', (), {})()
        
        # 测试零样本合成参数
        print("1. 测试零样本合成:")
        kwargs = adapter._prepare_generation_kwargs(
            text="今天天气很好",
            speaker_wav=None,
            language="zh"
        )
        print(f"   参数: {list(kwargs.keys())}")
        print(f"   持续时间: {kwargs.get('duration')}")
        
        # 测试语音克隆参数
        print("\n2. 测试语音克隆:")
        # 创建一个虚拟音频文件
        import tempfile
        import numpy as np
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # 创建1秒的静音音频
            audio_data = np.zeros(24000)  # 1秒@24kHz
            sf.write(f.name, audio_data, 24000)
            
            # 创建对应的文本文件
            txt_file = f.name.replace('.wav', '.txt')
            with open(txt_file, 'w', encoding='utf-8') as tf:
                tf.write("这是参考文本")
            
            try:
                kwargs = adapter._prepare_generation_kwargs(
                    text="今天天气很好",
                    speaker_wav=f.name,
                    language="zh"
                )
                print(f"   参数: {list(kwargs.keys())}")
                print(f"   是否有prompt_audio: {'prompt_audio' in kwargs}")
                print(f"   持续时间: {kwargs.get('duration')}")
            finally:
                # 清理临时文件
                os.unlink(f.name)
                if os.path.exists(txt_file):
                    os.unlink(txt_file)
        
        print("\n✅ 生成参数测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试简化版LongCat适配器...\n")
    
    tests = [
        ("适配器结构", test_adapter_structure),
        ("生成参数", test_generation_kwargs),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        result = test_func()
        results.append((test_name, result))
    
    # 总结
    print(f"\n{'='*50}")
    print("测试总结:")
    print('='*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过! 简化版适配器结构正确。")
        print("\n注意: 这是结构测试，实际模型加载和推理需要:")
        print("1. 安装LongCat-AudioDiT依赖")
        print("2. 下载模型权重")
        print("3. 有可用的GPU/CPU资源")
    else:
        print("\n⚠️  部分测试失败，请检查代码。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)