"""TTS适配器基类"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path


class BaseTTSAdapter(ABC):
    """TTS适配器基类，定义统一接口"""
    
    def __init__(self, model_repo: str):
        """
        初始化适配器
        
        Args:
            model_repo: HuggingFace模型仓库
        """
        self.model_repo = model_repo
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> Dict[str, str]:
        """
        获取支持的语言列表
        
        Returns:
            Dict[str, str]: 语言代码到语言名称的映射
        """
        pass
    
    @abstractmethod
    def get_tts_settings(self) -> Dict[str, Any]:
        """
        获取TTS设置
        
        Returns:
            Dict[str, Any]: TTS设置
        """
        pass
    
    @abstractmethod
    def update_tts_settings(self, settings: Dict[str, Any]):
        """
        更新TTS设置
        
        Args:
            settings: 新的TTS设置
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        pass
    
    def prepare_synthesis(
        self,
        speaker_wav: Optional[str],
        speaker_text: Optional[str] = None,
        design_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        准备合成参数
        
        Args:
            speaker_wav: 说话人音频路径
            speaker_text: 说话人参考文本（可选）
            design_description: 语音设计描述（可选）
            
        Returns:
            Dict[str, Any]: 合成参数
        """
        params = {}
        
        if speaker_wav and Path(speaker_wav).exists():
            params["ref_audio"] = speaker_wav
            if speaker_text:
                params["ref_text"] = speaker_text
            else:
                # 如果没有提供参考文本，可能需要自动转录
                params["ref_text"] = None
        
        if design_description:
            params["instruct"] = design_description
        
        return params
    
    def ensure_loaded(self):
        """确保模型已加载"""
        if not self.is_loaded:
            self.load_model()
            self.is_loaded = True
    
    def cleanup(self):
        """清理资源"""
        self.model = None
        self.is_loaded = False
    
    def __del__(self):
        """析构函数"""
        self.cleanup()