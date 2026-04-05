#!/usr/bin/env python3
"""测试TTS集成功能"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

print("="*60)
print("TTS集成功能测试")
print("="*60)

try:
    # 1. 测试模型管理器
    print("1. 测试模型管理器...")
    from models.manager import TTSModelManager
    
    # 测试OmniVoice
    manager_omni = TTSModelManager("k2-fsa/OmniVoice")
    print(f"✅ OmniVoice管理器创建成功: framework={manager_omni.framework}")
    
    # 测试LongCat-AudioDiT
    manager_longcat = TTSModelManager("meituan-longcat/LongCat-AudioDiT-1B")
    print(f"✅ LongCat-AudioDiT管理器创建成功: framework={manager_longcat.framework}")
    
    # 2. 测试模型加载
    print("\n2. 测试OmniVoice模型加载...")
    try:
        manager_omni.load_model()
        print("✅ OmniVoice模型加载成功")
        
        # 检查模型信息
        info = manager_omni.get_model_info()
        print(f"✅ 模型信息: {info}")
        
        # 检查支持的语言
        languages = manager_omni.get_supported_languages()
        print(f"✅ 支持的语言: {len(languages)} 种")
        
        # 检查TTS设置
        settings = manager_omni.get_tts_settings()
        print(f"✅ TTS设置: {settings}")
        
    except Exception as e:
        print(f"❌ OmniVoice模型加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 测试说话人扫描
    print("\n3. 测试说话人扫描...")
    from utils.file_utils import scan_speakers
    
    samples_dir = project_root / "samples"
    if samples_dir.exists():
        speakers = scan_speakers(samples_dir)
        print(f"✅ 扫描到 {len(speakers)} 个说话人")
        if speakers:
            for speaker in speakers:
                print(f"  - {speaker['id']}: {speaker['type']} ({speaker.get('path', 'N/A')})")
    else:
        print("⚠ Samples目录不存在")
    
    # 4. 测试TTS合成（如果有样本文件）
    print("\n4. 测试TTS合成...")
    paimon_wav = samples_dir / "paimon.wav"
    paimon_txt = samples_dir / "paimon.txt"
    
    if paimon_wav.exists() and paimon_txt.exists():
        print(f"✅ 找到示例文件: {paimon_wav.name}")
        
        # 读取文本
        with open(paimon_txt, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        print(f"✅ 使用文本: {text[:60]}...")
        
        # 尝试合成
        try:
            audio_data = manager_omni.synthesize(
                text=text,
                speaker_wav=str(paimon_wav),
                language='zh',
                speed=1.0
            )
            
            print(f"✅ TTS合成成功!")
            print(f"✅ 音频形状: {audio_data.shape}")
            print(f"✅ 音频长度: {len(audio_data)} 样本点")
            print(f"✅ 音频时长: {len(audio_data)/24000:.2f} 秒")
            
            # 保存测试输出
            output_dir = project_root / "output"
            output_dir.mkdir(exist_ok=True)
            
            import soundfile as sf
            output_path = output_dir / "test_output.wav"
            sf.write(output_path, audio_data, 24000)
            print(f"✅ 音频已保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ TTS合成失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠ 示例文件不完整，跳过TTS合成测试")
        print(f"  需要文件: {paimon_wav.name} 和 {paimon_txt.name}")
    
    # 5. 测试API端点
    print("\n5. 测试FastAPI应用...")
    try:
        from api.endpoints import create_app
        from config import get_default_config
        
        config = get_default_config()
        app = create_app(config)
        
        # 检查关键端点
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        print(f"✅ FastAPI应用创建成功，共 {len(routes)} 个路由")
        
        # 检查健康端点
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
        print(f"❌ API测试失败: {e}")
    
    print("\n" + "="*60)
    print("集成测试完成")
    print("="*60)
    
except Exception as e:
    print(f"❌ 集成测试失败: {e}")
    import traceback
    traceback.print_exc()