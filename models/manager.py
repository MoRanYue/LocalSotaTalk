"""TTS模型管理器"""
import importlib
from typing import Optional, Dict, Any
import numpy as np
try:
    from .base_adapter import BaseTTSAdapter
except ImportError:
    from models.base_adapter import BaseTTSAdapter


class TTSModelManager:
    """TTS模型管理器，负责加载和管理TTS模型"""
    
    def __init__(self, model_repo: str):
        """
        初始化模型管理器
        
        Args:
            model_repo: HuggingFace模型仓库
        """
        self.model_repo = model_repo
        self.framework = self._detect_framework(model_repo)
        self.adapter: Optional[BaseTTSAdapter] = None
        self.current_settings: Dict[str, Any] = {}
    
    def _detect_framework(self, repo: str) -> str:
        """
        根据模型仓库检测框架类型
        
        Args:
            repo: 模型仓库名称
            
        Returns:
            str: 框架类型 ('omnivoice', 'longcat' 或 'voxcpm')
        """
        repo_lower = repo.lower()
        if "voxcpm" in repo_lower:
            return "voxcpm"
        elif "omnivoice" in repo_lower:
            return "omnivoice"
        elif "longcat" in repo_lower or "audiodit" in repo_lower:
            return "longcat"
        else:
            # 默认尝试VoxCPM（目前最新/活跃的框架）
            return "voxcpm"
    
    def load_model(self) -> BaseTTSAdapter:
        """
        加载模型适配器
        
        Returns:
            BaseTTSAdapter: 加载的适配器
            
        Raises:
            ImportError: 如果无法导入相应的框架
            RuntimeError: 如果模型加载失败
        """
        if self.adapter is not None:
            return self.adapter
        
        try:
            if self.framework == "voxcpm":
                from .voxcpm_adapter import VoxCPMAdapter
                self.adapter = VoxCPMAdapter(self.model_repo)
            elif self.framework == "omnivoice":
                from .omnivoice_adapter import OmniVoiceAdapter
                self.adapter = OmniVoiceAdapter(self.model_repo)
            elif self.framework == "longcat":
                from .longcat_adapter import LongCatAdapter
                self.adapter = LongCatAdapter(self.model_repo)
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")
            
            # 加载模型
            self.adapter.load_model()
            
            # 获取默认设置
            self.current_settings = self.adapter.get_tts_settings()
            
            return self.adapter
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.framework} adapter. "
                f"Please ensure the framework is installed. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_repo}: {e}")
    
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
            
        Raises:
            RuntimeError: 如果模型未加载或合成失败
        """
        if self.adapter is None:
            self.load_model()
        
        # 此时adapter应该已加载
        adapter = self.adapter
        if adapter is None:
            raise RuntimeError("Failed to load adapter")
        
        try:
            # 应用当前设置
            synthesis_kwargs = {**self.current_settings, **kwargs}
            
            # 调用适配器合成
            audio = adapter.synthesize(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                **synthesis_kwargs
            )
            
            return audio
            
        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")
    
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
            
        Raises:
            RuntimeError: 如果模型未加载或合成失败
            NotImplementedError: 如果当前模型不支持音频设计合成
        """
        if self.adapter is None:
            self.load_model()
        
        # 此时adapter应该已加载
        adapter = self.adapter
        if adapter is None:
            raise RuntimeError("Failed to load adapter")
        
        try:
            # 应用当前设置
            synthesis_kwargs = {**self.current_settings, **kwargs}
            
            # 调用适配器的instructive合成方法
            audio = adapter.synthesize_instructively(
                text=text,
                design_description=design_description,
                language=language,
                **synthesis_kwargs
            )
            
            return audio
            
        except NotImplementedError:
            # 重新抛出NotImplementedError，让调用者知道这个功能不被支持
            raise
        except Exception as e:
            raise RuntimeError(f"Instructive synthesis failed: {e}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        获取支持的语言列表
        
        Returns:
            Dict[str, str]: 语言代码到语言名称的映射
        """
        if self.adapter is None:
            self.load_model()
        
        adapter = self.adapter
        if adapter is None:
            raise RuntimeError("Failed to load adapter")
        
        return adapter.get_supported_languages()
    
    def get_tts_settings(self) -> Dict[str, Any]:
        """
        获取当前TTS设置
        
        Returns:
            Dict[str, Any]: TTS设置
        """
        return self.current_settings.copy()
    
    def update_tts_settings(self, settings: Dict[str, Any]):
        """
        更新TTS设置
        
        Args:
            settings: 新的TTS设置
            
        Raises:
            ValueError: 如果设置无效
        """
        # 验证设置
        self._validate_settings(settings)
        
        # 更新当前设置
        self.current_settings.update(settings)
        
        # 如果适配器已加载，更新适配器设置
        if self.adapter is not None:
            self.adapter.update_tts_settings(self.current_settings)
    
    def _validate_settings(self, settings: Dict[str, Any]):
        """
        验证TTS设置
        
        Args:
            settings: 要验证的设置
            
        Raises:
            ValueError: 如果设置无效
        """
        # 这里可以添加具体的验证逻辑
        # 例如检查必填字段、值范围等
        
        # 示例验证
        if "temperature" in settings:
            temp = settings["temperature"]
            if not isinstance(temp, (int, float)):
                raise ValueError("Temperature must be a number")
            if temp < 0 or temp > 5:
                raise ValueError("Temperature must be between 0 and 5")
        
        if "speed" in settings:
            speed = settings["speed"]
            if not isinstance(speed, (int, float)):
                raise ValueError("Speed must be a number")
            if speed <= 0 or speed > 5:
                raise ValueError("Speed must be between 0.1 and 5")
    
    def get_framework_info(self) -> Dict[str, Any]:
        """
        获取框架信息
        
        Returns:
            Dict[str, Any]: 框架信息
        """
        return {
            "framework": self.framework,
            "model_repo": self.model_repo,
            "is_loaded": self.adapter is not None
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        if self.adapter is None:
            self.load_model()
        
        adapter = self.adapter
        if adapter is None:
            raise RuntimeError("Failed to load adapter")
        
        # 如果适配器有get_model_info方法，调用它
        if hasattr(adapter, 'get_model_info'):
            return adapter.get_model_info()
        
        # 否则返回基本信息
        return {
            "framework": self.framework,
            "model_repo": self.model_repo,
            "is_loaded": adapter.is_loaded if hasattr(adapter, 'is_loaded') else False
        }
    
    def cleanup(self):
        """清理资源"""
        if self.adapter is not None:
            self.adapter.cleanup()
            self.adapter = None
        self.current_settings = {}
    
    def reload_model(self, new_model_repo: Optional[str] = None):
        """
        重新加载模型
        
        Args:
            new_model_repo: 新的模型仓库（可选）
            
        Raises:
            RuntimeError: 如果重新加载失败
        """
        # 清理当前模型
        self.cleanup()
        
        # 更新模型仓库（如果提供）
        if new_model_repo:
            self.model_repo = new_model_repo
            self.framework = self._detect_framework(new_model_repo)
        
        # 重新加载
        self.load_model()