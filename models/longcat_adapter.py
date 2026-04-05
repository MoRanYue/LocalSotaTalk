"""LongCat-AudioDiT TTS适配器"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
import soundfile as sf

from models.base_adapter import BaseTTSAdapter
from utils.constants import LONGCAT_LANGUAGES, DEFAULT_TTS_SETTINGS

# 添加本地LongCat-AudioDiT模块路径
LOCAL_LONGCAT_PATH = Path(__file__).parent.parent / "systems" / "LongCat-AudioDiT"
if LOCAL_LONGCAT_PATH.exists():
    sys.path.insert(0, str(LOCAL_LONGCAT_PATH))


class LongCatAdapter(BaseTTSAdapter):
    """LongCat-AudioDiT TTS适配器"""
    
    def __init__(self, model_repo: str):
        super().__init__(model_repo)
        self.generation_config = {}
        self.sample_rate = 24000
        self.tokenizer = None
        self.vae = None
        
    def load_model(self):
        """加载LongCat-AudioDiT模型"""
        try:
            # 尝试导入LongCat-AudioDiT
            # 注意：LongCat-AudioDiT使用audiodit模块
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
            
            # 设置默认生成配置
            self.generation_config = {
                "steps": 16,           # 扩散步骤
                "cfg_strength": 4.0,   # CFG强度
                "guidance_method": "cfg",  # 引导方法
                "temperature": 1.0,
                "speed": 1.0,
            }
            
            self.is_loaded = True
            print("LongCat-AudioDiT model loaded successfully")
            
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
        language: str = "zh",  # LongCat主要支持中文
        **kwargs
    ) -> np.ndarray:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            speaker_wav: 说话人参考音频路径（必需）
            language: 语言代码（zh或en）
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 音频数据
        """
        self.ensure_loaded()
        
        if not speaker_wav:
            raise ValueError("LongCat-AudioDiT requires speaker_wav for voice cloning")
        
        try:
            # 准备生成参数
            gen_kwargs = self._prepare_generation_kwargs(text, speaker_wav, language, **kwargs)
            
            # 生成音频
            with torch.no_grad():
                output = self.model(**gen_kwargs)
            
            # 提取音频数据
            if hasattr(output, 'waveform'):
                audio_tensor = output.waveform
            elif hasattr(output, 'audio'):
                audio_tensor = output.audio
            else:
                raise ValueError("Model output does not contain audio data")
            
            # 转换为numpy数组
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.squeeze().cpu().numpy()
            else:
                audio_np = np.array(audio_tensor).squeeze()
            
            return audio_np
            
        except Exception as e:
            raise RuntimeError(f"LongCat-AudioDiT synthesis failed: {e}")
    
    def _prepare_generation_kwargs(
        self,
        text: str,
        speaker_wav: str,
        language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备生成参数
        
        Args:
            text: 合成文本
            speaker_wav: 说话人音频路径
            language: 语言代码
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 生成参数
        """
        import librosa
        
        gen_kwargs = self.generation_config.copy()
        gen_kwargs.update(kwargs)
        
        # 读取参考音频
        if not Path(speaker_wav).exists():
            raise FileNotFoundError(f"Speaker audio file not found: {speaker_wav}")
        
        # 加载音频
        audio, sr = librosa.load(speaker_wav, sr=24000, mono=True)
        prompt_wav = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        
        # 查找参考文本
        txt_file = Path(speaker_wav).with_suffix(".txt")
        prompt_text = ""
        if txt_file.exists():
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()
            except Exception:
                warnings.warn(f"Failed to read text file {txt_file}")
        
        # LongCat需要prompt_text + gen_text拼接
        if prompt_text:
            full_text = f"{prompt_text} {text}"
        else:
            # 如果没有参考文本，只使用生成文本
            full_text = text
            warnings.warn(f"No reference text found for {speaker_wav}, using text-only mode")
        
        # Tokenize文本
        inputs = self.tokenizer([full_text], padding="longest", return_tensors="pt")
        
        # 估计持续时间（简化版本）
        # 在实际实现中，可能需要更准确的持续时间估计
        duration = self._estimate_duration(text, language)
        
        # 构建生成参数
        result = {
            "input_ids": inputs.input_ids.to(self.model.device),
            "attention_mask": inputs.attention_mask.to(self.model.device) if hasattr(inputs, 'attention_mask') else None,
            "prompt_audio": prompt_wav.to(self.model.device),
            "duration": duration,
        }
        
        # 添加其他生成参数
        result.update({
            k: v for k, v in gen_kwargs.items()
            if k in ["steps", "cfg_strength", "guidance_method", "temperature", "speed"]
        })
        
        return result
    
    def _estimate_duration(self, text: str, language: str) -> int:
        """
        估计音频持续时间（帧数）
        
        Args:
            text: 文本
            language: 语言
            
        Returns:
            int: 估计的帧数
        """
        # 简化估计：假设每个字符对应一定帧数
        # LongCat使用24000Hz采样率，hop_size=256
        hop_size = 256
        
        if language == "zh":
            # 中文：每个字符约0.3秒
            chars_per_second = 3.3
        else:
            # 英文：每个字符约0.08秒
            chars_per_second = 12.5
        
        duration_seconds = len(text) / chars_per_second
        frames = int(duration_seconds * 24000 / hop_size)
        
        # 确保最小帧数
        return max(frames, 10)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        获取支持的语言列表
        
        Returns:
            Dict[str, str]: 语言代码到语言名称的映射
        """
        return LONGCAT_LANGUAGES.copy()
    
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
        valid_keys = ["steps", "cfg_strength", "guidance_method", "temperature", "speed"]
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
        
        Args:
            text: 要合成的文本
            design_description: 音频设计描述文本
            language: 语言代码
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 音频数据
        """
        raise NotImplementedError(
            "LongCat-AudioDiT does not support instructive synthesis. "
            "Please use other models for voice design functionality."
        )
