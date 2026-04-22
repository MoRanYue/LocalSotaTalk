"""VoxCPM TTS适配器"""
import sys
import os
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

from models.base_adapter import BaseTTSAdapter
from utils.constants import VOXCPM_LANGUAGES, DEFAULT_TTS_SETTINGS

# 添加本地VoxCPM模块路径（如果存在）
LOCAL_VOXCPM_PATH = Path(__file__).parent.parent / "systems" / "VoxCPM"
if LOCAL_VOXCPM_PATH.exists():
    # 添加src目录到sys.path，因为VoxCPM包在src/voxcpm中
    src_path = LOCAL_VOXCPM_PATH / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    # 同时添加根目录作为备用
    sys.path.insert(0, str(LOCAL_VOXCPM_PATH))


class VoxCPMAdapter(BaseTTSAdapter):
    """VoxCPM TTS适配器"""
    
    def __init__(self, model_repo: str):
        super().__init__(model_repo)
        self.generation_config = {}
        self.sample_rate = 48000  # VoxCPM2默认48kHz，加载后从模型获取
        self._voxcpm_instance = None  # 保存VoxCPM实例引用
    
    def _import_voxcpm(self):
        """
        导入VoxCPM类
        
        Returns:
            VoxCPM类
        """
        try:
            from voxcpm import VoxCPM
            return VoxCPM
        except ImportError:
            # 如果标准导入失败，尝试从本地路径导入
            local_voxcpm_path = Path(__file__).parent.parent / "systems" / "VoxCPM" / "src"
            if local_voxcpm_path.exists():
                import sys
                sys.path.insert(0, str(local_voxcpm_path.parent.parent))  # 添加根目录
                sys.path.insert(0, str(local_voxcpm_path))  # 添加src目录
                from voxcpm import VoxCPM
                return VoxCPM
            else:
                raise ImportError(
                    "VoxCPM is not installed. Please install it via: pip install voxcpm\n"
                    "Or ensure the VoxCPM directory exists in systems/VoxCPM"
                )
        
    def load_model(self):
        """加载VoxCPM模型"""
        try:
            # 延迟导入VoxCPM，避免静态分析错误
            VoxCPM = self._import_voxcpm()
            
            # 设置设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"Loading VoxCPM model from {self.model_repo}...")
            
            # 加载模型
            # 支持本地路径和HuggingFace repo ID
            if os.path.isdir(self.model_repo):
                # 本地路径：直接使用VoxCPM构造函数
                self._voxcpm_instance = VoxCPM(
                    voxcpm_model_path=self.model_repo,
                    enable_denoiser=False,
                    optimize=True,
                    device=device,
                )
            else:
                # HuggingFace repo ID
                self._voxcpm_instance = VoxCPM.from_pretrained(
                    hf_model_id=self.model_repo,
                    load_denoiser=False,
                    optimize=True,
                    device=device,
                )
            
            self.model = self._voxcpm_instance
            
            # 从模型获取实际采样率
            # VoxCPM模型可能有多个采样率属性：
            # 1. tts_model.sample_rate - 输出采样率
            # 2. tts_model._encode_sample_rate - 编码输入采样率
            # 3. 对于VoxCPM v2，还有out_sample_rate属性
            if hasattr(self.model.tts_model, 'sample_rate'):
                self.sample_rate = self.model.tts_model.sample_rate
                print(f"Detected sample rate: {self.sample_rate}Hz")
            elif hasattr(self.model.tts_model, '_encode_sample_rate'):
                # 如果有编码采样率，尝试获取输出采样率
                if hasattr(self.model.tts_model, 'out_sample_rate'):
                    self.sample_rate = self.model.tts_model.out_sample_rate
                else:
                    self.sample_rate = self.model.tts_model._encode_sample_rate
                print(f"Detected sample rate from encode rate: {self.sample_rate}Hz")
            else:
                # 默认使用48kHz（VoxCPM2标准输出）
                self.sample_rate = 48000
                print(f"Using default sample rate: {self.sample_rate}Hz")
            
            # 检测是否为VoxCPM2模型
            # VoxCPM2支持reference_wav_path参数，VoxCPM1不支持
            self.is_voxcpm2 = (
                hasattr(self.model.tts_model, 'out_sample_rate') or 
                self.sample_rate == 48000 or
                'voxcpm2' in str(type(self.model.tts_model)).lower()
            )
            print(f"Model type: {'VoxCPM2' if self.is_voxcpm2 else 'VoxCPM1'}")
            
            # 设置默认生成配置
            self.generation_config = {
                "cfg_value": 2.0,           # CFG引导强度
                "inference_timesteps": 10,   # 扩散步数
                "normalize": False,          # 文本规范化
                "denoise": False,            # 参考音频降噪
            }
            
            self.is_loaded = True
            print(f"VoxCPM model loaded successfully on {device}, sample_rate={self.sample_rate}, is_voxcpm2={self.is_voxcpm2}")
            
        except ImportError as e:
            raise ImportError(
                "VoxCPM is not installed. Please install it via: pip install voxcpm\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load VoxCPM model: {e}")
    
    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> np.ndarray:
        """
        合成语音
        
        VoxCPM支持三种模式：
        1. 零样本（无参考音频）：直接TTS
        2. 可控克隆（有参考音频）：克隆音色+可选的风格控制
        3. 极致克隆（有参考音频+文本）：音频续写方式高保真克隆
        
        Args:
            text: 要合成的文本
            speaker_wav: 说话人参考音频路径（可选）
            language: 语言代码
            **kwargs: 其他参数
                - control_instruction: 控制指令（描述音色/风格）
                - prompt_text: 参考音频对应的文本内容（用于极致克隆）
                - cfg_value: CFG引导强度
                - inference_timesteps: 扩散步数
                - normalize: 文本规范化
                - denoise: 参考音频降噪
                
        Returns:
            np.ndarray: 音频数据
        """
        self.ensure_loaded()
        
        try:
            # 准备生成参数，先过滤掉VoxCPM不支持的参数
            filtered_kwargs = self._filter_voxcpm_kwargs(kwargs)
            gen_kwargs = self._prepare_generation_kwargs(speaker_wav, **filtered_kwargs)
            
            # 构建最终文本
            # VoxCPM支持在文本前用括号包裹音色描述：(描述)文本
            final_text = text
            control_instruction = gen_kwargs.pop("control_instruction", None)
            if control_instruction:
                # 去除可能影响格式的括号字符
                control = re.sub(r"[()（）]", "", control_instruction).strip()
                if control:
                    final_text = f"({control}){text}"
            
            # 生成音频
            wav = self._voxcpm_instance.generate(
                text=final_text,
                **gen_kwargs
            )
            
            # 确保返回numpy数组
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            
            # 确保是单声道1维数组
            if len(wav.shape) > 1:
                wav = wav.squeeze()
            
            return wav
            
        except Exception as e:
            raise RuntimeError(f"VoxCPM synthesis failed: {e}")
    
    def _prepare_generation_kwargs(
        self,
        speaker_wav: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备生成参数
        
        Args:
            speaker_wav: 说话人音频路径
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 生成参数
        """
        gen_kwargs = self.generation_config.copy()
        
        # VoxCPM的generate方法支持的参数列表
        # 参考: core.py中的_generate方法参数
        voxcpm_supported_params = {
            "control_instruction",  # 控制指令
            "prompt_text",          # 参考音频文本
            "prompt_wav_path",      # 参考音频路径（用于极致克隆）
            "reference_wav_path",   # 参考音频路径（用于音色克隆）
            "cfg_value",            # CFG引导强度
            "inference_timesteps",  # 扩散步数
            "min_len",              # 最小长度
            "max_len",              # 最大长度
            "normalize",            # 文本规范化
            "denoise",              # 参考音频降噪
            "retry_badcase",        # 重试错误案例
            "retry_badcase_max_times",  # 最大重试次数
            "retry_badcase_ratio_threshold",  # 重试阈值
        }
        
        # 只传递VoxCPM支持的参数
        for key in voxcpm_supported_params:
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]
        
        # 处理说话人音频
        prompt_text = gen_kwargs.get("prompt_text", None)
        
        if speaker_wav and Path(speaker_wav).exists():
            # 检查是否为语音设计文件 (.design.txt)
            if speaker_wav.endswith(".design.txt"):
                try:
                    with open(speaker_wav, 'r', encoding='utf-8') as f:
                        design_desc = f.read().strip()
                    gen_kwargs["control_instruction"] = design_desc
                except Exception as e:
                    warnings.warn(f"Failed to read design file {speaker_wav}: {e}")
            else:
                # 语音克隆模式
                # VoxCPM2支持reference_wav_path参数，VoxCPM1不支持
                if self.is_voxcpm2:
                    gen_kwargs["reference_wav_path"] = speaker_wav
                
                # 如果有prompt_text，启用极致克隆模式
                if prompt_text:
                    # VoxCPM的极致克隆：传递prompt_wav_path
                    gen_kwargs["prompt_wav_path"] = speaker_wav
                else:
                    # 尝试查找对应的文本文件用于极致克隆
                    txt_file = Path(speaker_wav).with_suffix(".txt")
                    if txt_file.exists():
                        try:
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                ref_text = f.read().strip()
                            gen_kwargs["prompt_wav_path"] = speaker_wav
                            gen_kwargs["prompt_text"] = ref_text
                        except Exception:
                            # 如果没有文本文件，对于VoxCPM1可能不支持语音克隆
                            # 对于VoxCPM2，我们已经设置了reference_wav_path
                            pass
        
        # 最终安全过滤：确保只返回VoxCPM支持的参数
        supported_keys = {
            "control_instruction", "prompt_text", "prompt_wav_path", "reference_wav_path",
            "cfg_value", "inference_timesteps", "min_len", "max_len", "normalize", "denoise",
            "retry_badcase", "retry_badcase_max_times", "retry_badcase_ratio_threshold"
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in supported_keys}
        
        return gen_kwargs
    
    def _filter_voxcpm_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤掉VoxCPM不支持的参数
        
        Args:
            kwargs: 原始参数
            
        Returns:
            Dict[str, Any]: 过滤后的参数
        """
        # VoxCPM的generate方法支持的参数列表
        voxcpm_supported_params = {
            "control_instruction",  # 控制指令
            "prompt_text",          # 参考音频文本
            "prompt_wav_path",      # 参考音频路径（用于极致克隆）
            "reference_wav_path",   # 参考音频路径（用于音色克隆）
            "cfg_value",            # CFG引导强度
            "inference_timesteps",  # 扩散步数
            "min_len",              # 最小长度
            "max_len",              # 最大长度
            "normalize",            # 文本规范化
            "denoise",              # 参考音频降噪
            "retry_badcase",        # 重试错误案例
            "retry_badcase_max_times",  # 最大重试次数
            "retry_badcase_ratio_threshold",  # 重试阈值
        }
        
        # 只保留VoxCPM支持的参数
        filtered = {}
        for key in voxcpm_supported_params:
            if key in kwargs:
                filtered[key] = kwargs[key]
        
        # 同时包含generation_config中的默认参数，但只保留支持的参数
        for key in voxcpm_supported_params:
            if key in self.generation_config:
                filtered[key] = self.generation_config[key]
        
        return filtered
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        获取支持的语言列表
        
        VoxCPM2支持30种语言，V1.5/0.5B仅支持中英双语
        
        Returns:
            Dict[str, str]: 语言代码到语言名称的映射
        """
        return VOXCPM_LANGUAGES.copy()
    
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
        # VoxCPM支持的参数列表
        voxcpm_supported_params = {
            "cfg_value",            # CFG引导强度
            "inference_timesteps",  # 扩散步数
            "normalize",            # 文本规范化
            "denoise",              # 参考音频降噪
            "control_instruction",  # 控制指令
            "prompt_text",          # 参考音频文本
            "prompt_wav_path",      # 参考音频路径（用于极致克隆）
            "reference_wav_path",   # 参考音频路径（用于音色克隆）
            "min_len",              # 最小长度
            "max_len",              # 最大长度
            "retry_badcase",        # 重试错误案例
            "retry_badcase_max_times",  # 最大重试次数
            "retry_badcase_ratio_threshold",  # 重试阈值
        }
        
        # 只更新VoxCPM支持的参数到generation_config
        for key in voxcpm_supported_params:
            if key in settings:
                self.generation_config[key] = settings[key]
        
        # 不支持的参数（如stream_chunk_size, temperature等）不保存到generation_config
        # 这些参数会在get_tts_settings中从DEFAULT_TTS_SETTINGS获取
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "framework": "voxcpm",
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
        通过音频设计描述合成语音（音色设计）
        
        VoxCPM原生支持音色设计：将设计描述作为控制指令放在文本前
        格式: "(音色描述)要合成的文本"
        
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
            # 准备生成参数，过滤掉VoxCPM不支持的参数
            gen_kwargs = self._filter_voxcpm_kwargs(kwargs)
            
            # VoxCPM音色设计：将描述包装为(描述)文本格式
            # 去除可能影响格式的括号字符
            description = re.sub(r"[()（）]", "", design_description).strip()
            if description:
                final_text = f"({description}){text}"
            else:
                final_text = text
            
            # 生成音频（音色设计模式不需要参考音频）
            wav = self._voxcpm_instance.generate(text=final_text, **gen_kwargs)
            
            # 确保返回numpy数组
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            
            # 确保是单声道1维数组
            if len(wav.shape) > 1:
                wav = wav.squeeze()
            
            return wav
            
        except Exception as e:
            raise RuntimeError(f"VoxCPM instructive synthesis failed: {e}")
    
    def cleanup(self):
        """清理资源"""
        self._voxcpm_instance = None
        self.model = None
        self.is_loaded = False
    
    def __del__(self):
        """析构函数"""
        self.cleanup()