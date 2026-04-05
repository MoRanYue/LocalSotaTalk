"""OmniVoice TTS适配器"""
import sys
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings

from models.base_adapter import BaseTTSAdapter
from utils.constants import OMNIVOICE_LANGUAGES, DEFAULT_TTS_SETTINGS

# 添加本地OmniVoice模块路径
LOCAL_OMNIVOICE_PATH = Path(__file__).parent.parent / "systems" / "OmniVoice"
if LOCAL_OMNIVOICE_PATH.exists():
    sys.path.insert(0, str(LOCAL_OMNIVOICE_PATH))


class OmniVoiceAdapter(BaseTTSAdapter):
    """OmniVoice TTS适配器"""
    
    def __init__(self, model_repo: str):
        super().__init__(model_repo)
        self.generation_config = {}
        self.sample_rate = 24000
        
    def load_model(self):
        """Load OmniVoice model"""
        try:
            # 尝试从本地子模块导入OmniVoice
            # 由于我们已经添加了LOCAL_OMNIVOICE_PATH到sys.path
            # 可以直接从omnivoice.models.omnivoice导入
            try:
                from omnivoice.models.omnivoice import OmniVoice
            except ImportError:
                # 回退到直接导入
                from omnivoice import OmniVoice
            
            # Set device - always use float32 for compatibility
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float32  # Always use float32 to avoid dtype mismatch
            
            print(f"Loading OmniVoice model from {self.model_repo}...")
            
            # 尝试多种加载方式，因为OmniVoice内部可能使用device_map
            try:
                # 方式1: 尝试明确设置device_map=None
                self.model = OmniVoice.from_pretrained(
                    self.model_repo,
                    device_map=None,  # 明确设置device_map=None
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                print("✅ 使用device_map=None加载成功")
            except Exception as e1:
                if "accelerate" in str(e1).lower() or "device_map" in str(e1).lower():
                    # 方式2: 尝试完全避免device_map相关参数
                    try:
                        self.model = OmniVoice.from_pretrained(
                            self.model_repo,
                            trust_remote_code=True
                        )
                        print("✅ 使用默认参数加载成功")
                    except Exception as e2:
                        # 方式3: 尝试通过加载到CPU然后移动到设备
                        try:
                            # 强制使用CPU加载
                            self.model = OmniVoice.from_pretrained(
                                self.model_repo,
                                device_map=None,
                                torch_dtype=torch.float32,  # CPU使用float32
                                trust_remote_code=True
                            )
                            print("✅ 使用CPU加载成功，准备移动到设备")
                        except Exception as e3:
                            raise RuntimeError(f"所有加载方式都失败: {e3}")
            
            # 确保模型在正确的设备上，保持float32以避免数据类型不匹配
            if self.model.device.type != device:
                print(f"将模型从 {self.model.device} 移动到 {device}")
                self.model = self.model.to(device, dtype=torch.float32)
            
            # Set default generation config
            self.generation_config = {
                "num_step": 32,  # Diffusion steps
                "speed": 1.0,    # Speed
                "temperature": 1.0,
            }
            
            self.is_loaded = True
            print(f"OmniVoice model loaded successfully on {device}")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import OmniVoice from local submodule: {e}\n"
                "Please ensure the OmniVoice submodule is initialized in systems/OmniVoice"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load OmniVoice model: {e}")
    
    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> np.ndarray:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            speaker_wav: 说话人参考音频路径（可选）
            language: 语言代码
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 音频数据
        """
        self.ensure_loaded()
        
        try:
            # 准备生成参数
            gen_kwargs = self._prepare_generation_kwargs(speaker_wav, language, **kwargs)
            
            # 生成音频
            with torch.no_grad():
                audio_list = self.model.generate(text=text, **gen_kwargs)
            
            # 提取音频数据
            if not audio_list:
                raise ValueError("No audio generated")
            
            # OmniVoice返回的是列表，每个元素是torch.Tensor
            audio_tensor = audio_list[0]  # 取第一个结果
            
            # 转换为numpy数组
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = np.array(audio_tensor)
            
            # 确保是单声道
            if len(audio_np.shape) > 1:
                audio_np = audio_np.squeeze()
            
            return audio_np
            
        except Exception as e:
            raise RuntimeError(f"OmniVoice synthesis failed: {e}")
    
    def _prepare_generation_kwargs(
        self,
        speaker_wav: Optional[str],
        language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备生成参数
        
        Args:
            speaker_wav: 说话人音频路径
            language: 语言代码
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 生成参数
        """
        gen_kwargs = self.generation_config.copy()
        gen_kwargs.update(kwargs)
        
        # 处理说话人信息
        if speaker_wav and Path(speaker_wav).exists():
            # 检查是否为语音设计文件
            if speaker_wav.endswith(".design.txt"):
                # 读取设计描述
                try:
                    with open(speaker_wav, 'r', encoding='utf-8') as f:
                        design_desc = f.read().strip()
                    gen_kwargs["instruct"] = design_desc
                except Exception as e:
                    warnings.warn(f"Failed to read design file {speaker_wav}: {e}")
            else:
                # 语音克隆模式
                gen_kwargs["ref_audio"] = speaker_wav
                # 尝试查找对应的文本文件
                txt_file = Path(speaker_wav).with_suffix(".txt")
                if txt_file.exists():
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            ref_text = f.read().strip()
                        gen_kwargs["ref_text"] = ref_text
                    except Exception:
                        # 如果没有文本文件，让模型自动转录
                        gen_kwargs["ref_text"] = None
        
        # 处理语言
        if language and language != "en":
            # OmniVoice支持通过文本前缀指定语言
            # 这里可以根据需要添加语言处理逻辑
            pass
        
        return gen_kwargs
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        获取支持的语言列表
        
        Returns:
            Dict[str, str]: 语言代码到语言名称的映射
        """
        return OMNIVOICE_LANGUAGES.copy()
    
    def get_tts_settings(self) -> Dict[str, Any]:
        """
        获取TTS设置
        
        Returns:
            Dict[str, Any]: TTS设置
        """
        settings = DEFAULT_TTS_SETTINGS.copy()
        settings.update(self.generation_config)
        return settings
    
    def update_tts_settings(self, settings: Dict[str, Any]):
        """
        更新TTS设置
        
        Args:
            settings: 新的TTS设置
        """
        # 更新生成配置
        valid_keys = ["num_step", "speed", "temperature", "duration", "top_p", "top_k"]
        for key in valid_keys:
            if key in settings:
                self.generation_config[key] = settings[key]
        
        # 更新其他设置
        self.generation_config.update({
            k: v for k, v in settings.items() 
            if k not in valid_keys
        })
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "framework": "omnivoice",
            "model_repo": self.model_repo,
            "sample_rate": self.sample_rate,
            "generation_config": self.generation_config.copy()
        }
    
    def synthesize_instructively(
        self,
        text: str,
        design_description: str,
        language: str = "en",
        **kwargs
    ) -> np.ndarray:
        """
        通过音频设计描述合成语音
        
        Args:
            text: 要合成的文本
            design_description: 音频设计描述文本
            language: 语言代码
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 音频数据
        """
        self.ensure_loaded()
        
        try:
            # 准备生成参数
            gen_kwargs = self.generation_config.copy()
            gen_kwargs.update(kwargs)
            
            # 设置语言
            if language and language != "en":
                # OmniVoice支持通过文本前缀指定语言
                # 这里可以根据需要添加语言处理逻辑
                pass
            
            # 使用音频设计描述进行生成
            with torch.no_grad():
                # OmniVoice的generate方法应该支持通过设计描述生成
                # 这里假设模型支持instruct参数
                audio_list = self.model.generate(
                    text=text,
                    instruct=design_description,
                    **gen_kwargs
                )
            
            # 提取音频数据
            if not audio_list:
                raise ValueError("No audio generated")
            
            # OmniVoice返回的是列表，每个元素是torch.Tensor
            audio_tensor = audio_list[0]  # 取第一个结果
            
            # 转换为numpy数组
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = np.array(audio_tensor)
            
            # 确保是单声道
            if len(audio_np.shape) > 1:
                audio_np = audio_np.squeeze()
            
            return audio_np
            
        except Exception as e:
            raise RuntimeError(f"OmniVoice instructive synthesis failed: {e}")
