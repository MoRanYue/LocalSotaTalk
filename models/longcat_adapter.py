"""LongCat-AudioDiT TTS适配器 - 简化版本"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
import librosa

from models.base_adapter import BaseTTSAdapter
from utils.constants import LONGCAT_LANGUAGES, DEFAULT_TTS_SETTINGS

# 添加本地LongCat-AudioDiT模块路径
LOCAL_LONGCAT_PATH = Path(__file__).parent.parent / "systems" / "LongCat-AudioDiT"
if LOCAL_LONGCAT_PATH.exists():
    sys.path.insert(0, str(LOCAL_LONGCAT_PATH))


class LongCatAdapter(BaseTTSAdapter):
    """LongCat-AudioDiT TTS适配器 - 简化版本"""
    
    def __init__(self, model_repo: str):
        super().__init__(model_repo)
        self.generation_config = {}
        self.sample_rate = 24000
        self.tokenizer = None
        
    def load_model(self):
        """加载LongCat-AudioDiT模型"""
        try:
            # 尝试导入LongCat-AudioDiT
            import audiodit
            from audiodit import AudioDiTModel
            from transformers import AutoTokenizer
            
            # 设置设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 加载模型
            print(f"Loading LongCat-AudioDiT model from {self.model_repo}...")
            self.model = AudioDiTModel.from_pretrained(
                self.model_repo,
                trust_remote_code=True
            ).to(device)
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder_model
            )
            
            # VAE使用半精度（匹配原始实现）
            if hasattr(self.model, 'vae'):
                self.model.vae.to(torch.float16)
            
            # 设置为评估模式
            self.model.eval()
            
            # 从模型配置获取采样率
            if hasattr(self.model.config, 'sampling_rate'):
                self.sample_rate = self.model.config.sampling_rate
                print(f"Model sampling rate: {self.sample_rate}Hz")
            else:
                print(f"Using default sampling rate: {self.sample_rate}Hz")
            
            # 设置默认生成配置（使用官方默认参数）
            self.generation_config = {
                "steps": 16,           # 扩散步骤（官方默认：nfe=16）
                "cfg_strength": 4.0,   # CFG强度（官方默认：guidance_strength=4.0）
                "guidance_method": "cfg",  # 引导方法（官方默认："cfg"）
            }
            
            self.is_loaded = True
            print(f"LongCat-AudioDiT model loaded successfully (sample_rate={self.sample_rate}Hz)")
            
        except ImportError as e:
            raise ImportError(
                "LongCat-AudioDiT is not installed. Please ensure the framework is available. "
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LongCat-AudioDiT model: {e}")
    
    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "zh",
        **kwargs
    ) -> np.ndarray:
        """
        合成语音 - 简化版本
        
        Args:
            text: 要合成的文本
            speaker_wav: 说话人参考音频路径（语音克隆必需）
            language: 语言代码（zh或en）
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 音频数据
        """
        self.ensure_loaded()
        
        try:
            # 准备生成参数
            gen_kwargs = self._prepare_generation_kwargs(text, speaker_wav, language, **kwargs)
            
            # 生成音频
            with torch.no_grad():
                output = self.model(**gen_kwargs)
            
            # 提取音频数据（根据官方示例）
            if hasattr(output, 'waveform'):
                audio_tensor = output.waveform
            else:
                # 如果返回元组，取第一个元素
                audio_tensor = output[0] if isinstance(output, tuple) else output
            
            # 转换为numpy数组
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = np.array(audio_tensor)
            
            # 确保是单声道1维数组
            audio_np = audio_np.squeeze()
            
            # 简单归一化防止爆音
            max_val = np.max(np.abs(audio_np))
            if max_val > 1.0:
                audio_np = audio_np / max_val
            
            return audio_np
            
        except Exception as e:
            raise RuntimeError(f"LongCat-AudioDiT synthesis failed: {e}")
    
    def _prepare_generation_kwargs(
        self,
        text: str,
        speaker_wav: Optional[str],
        language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备生成参数 - 简化版本
        
        根据官方README实现：
        1. 零样本合成：只需要text
        2. 语音克隆：需要speaker_wav和prompt_text
        """
        import torch.nn.functional as F
        
        # 合并配置
        gen_kwargs = self.generation_config.copy()
        gen_kwargs.update(kwargs)
        
        # 处理文本
        prompt_text = ""
        full_text = text
        
        # 如果有参考音频，尝试读取对应的文本文件
        if speaker_wav and Path(speaker_wav).exists():
            txt_file = Path(speaker_wav).with_suffix(".txt")
            if txt_file.exists():
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        prompt_text = f.read().strip()
                except Exception:
                    pass
            
            # LongCat需要prompt_text + gen_text拼接
            if prompt_text:
                full_text = f"{prompt_text} {text}"
        
        # Tokenize文本
        inputs = self.tokenizer([full_text], padding="longest", return_tensors="pt")
        
        # 构建基础参数
        result = {
            "input_ids": inputs.input_ids.to(self.model.device),
            "attention_mask": inputs.attention_mask.to(self.model.device) if hasattr(inputs, 'attention_mask') else None,
            "return_dict": True,  # 确保返回AudioDiTOutput对象
        }
        
        # 添加参考音频（如果提供）
        if speaker_wav and Path(speaker_wav).exists():
            # 加载音频（使用模型采样率）
            audio, _ = librosa.load(speaker_wav, sr=self.sample_rate, mono=True)
            prompt_wav = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            result["prompt_audio"] = prompt_wav.to(self.model.device)
        
        # 估计持续时间（简化版本）
        # 官方示例：零样本合成使用62帧，语音克隆使用138帧
        # 这里根据文本长度简单估计
        duration = self._estimate_duration(text, language, speaker_wav is not None)
        result["duration"] = duration
        
        # 添加生成参数
        result.update({
            "steps": gen_kwargs.get("steps", 16),
            "cfg_strength": gen_kwargs.get("cfg_strength", 4.0),
            "guidance_method": gen_kwargs.get("guidance_method", "cfg"),
        })
        
        return result
    
    def _estimate_duration(self, text: str, language: str, has_prompt: bool) -> int:
        """
        估计持续时间（潜在帧数） - 简化版本
        
        这里根据文本长度简单估计
        """
        # 根据文本长度调整
        # 假设每个中文字符约0.3秒，每个英文字符约0.08秒
        if language == "zh":
            chars_per_second = 3.3
        else:
            chars_per_second = 12.5
        
        duration_seconds = len(text) / chars_per_second
        
        # 转换为帧数（假设hop_size=256，采样率24000）
        # 每秒帧数 = 采样率 / hop_size = 24000 / 256 ≈ 93.75
        frames_per_second = self.sample_rate / 256
        total_frames = int(duration_seconds * frames_per_second)
        
        # 限制范围
        return max(50, min(total_frames, 300))
    
    def get_supported_languages(self) -> Dict[str, str]:
        """获取支持的语言列表"""
        return LONGCAT_LANGUAGES.copy()
    
    def get_tts_settings(self) -> Dict[str, Any]:
        """获取TTS设置"""
        settings = DEFAULT_TTS_SETTINGS.copy()
        settings.update(self.generation_config)
        return settings
    
    def update_tts_settings(self, settings: Dict[str, Any]):
        """更新TTS设置"""
        valid_keys = ["steps", "cfg_strength", "guidance_method"]
        for key in valid_keys:
            if key in settings:
                self.generation_config[key] = settings[key]
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "framework": "longcat",
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
        LongCat-AudioDiT不支持音频设计，抛出异常
        """
        raise NotImplementedError(
            "LongCat-AudioDiT does not support instructive synthesis. "
            "Please use other models for voice design functionality."
        )