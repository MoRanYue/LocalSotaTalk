"""TTS后端API端点实现"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
from datetime import datetime
import uuid

# 使用相对导入时注意模块层级
try:
    from .schemas import (
        ModelNameRequest, OutputFolderRequest, SpeakerFolderRequest,
        SynthesisRequest, SynthesisFileRequest, TTSSettingsRequest,
        SpeakerInfo, LanguageInfo, ModelInfo, FolderInfo, TTSSettingsInfo,
        SynthesisResponse, StatusResponse, SpeakerListResponse,
        LanguageListResponse, ModelListResponse, create_error_response
    )
    from ..models.manager import TTSModelManager
    from ..utils.file_utils import scan_speakers, save_audio_file, get_speaker_by_id, read_audio_file
    from ..utils.constants import SUPPORTED_LANGUAGES, DEFAULT_TTS_SETTINGS
    from ..config import AppConfig, get_default_config
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from api.schemas import (
        ModelNameRequest, OutputFolderRequest, SpeakerFolderRequest,
        SynthesisRequest, SynthesisFileRequest, TTSSettingsRequest,
        SpeakerInfo, LanguageInfo, ModelInfo, FolderInfo, TTSSettingsInfo,
        SynthesisResponse, StatusResponse, SpeakerListResponse,
        LanguageListResponse, ModelListResponse, create_error_response
    )
    from models.manager import TTSModelManager
    from utils.file_utils import scan_speakers, save_audio_file, get_speaker_by_id, read_audio_file
    from utils.constants import SUPPORTED_LANGUAGES, DEFAULT_TTS_SETTINGS
    from config import AppConfig, get_default_config


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSAPI:
    """TTS API实现类"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.model_manager: Optional[TTSModelManager] = None
        self.app = FastAPI(
            title="TTS Backend Service",
            description="支持LongCat-AudioDiT和OmniVoice的TTS后端服务",
            version="0.1.0"
        )
        self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        # 管理端点
        self.app.get("/speakers_list")(self.get_speakers_list)
        self.app.get("/speakers")(self.get_speakers)
        self.app.get("/languages")(self.get_languages)
        self.app.get("/get_folders")(self.get_folders)
        self.app.get("/get_models_list")(self.get_models_list)
        self.app.get("/get_tts_settings")(self.get_tts_settings)
        self.app.get("/sample/{file_name}")(self.get_sample)
        
        # 配置端点
        self.app.post("/set_output")(self.set_output_folder)
        self.app.post("/set_speaker_folder")(self.set_speaker_folder)
        self.app.post("/switch_model")(self.switch_model)
        self.app.post("/set_tts_settings")(self.set_tts_settings)
        
        # TTS生成端点
        self.app.get("/tts_stream")(self.tts_stream)
        self.app.post("/tts_to_audio/")(self.tts_to_audio)
        self.app.post("/tts_to_file")(self.tts_to_file)
        
        # 健康检查
        self.app.get("/health")(self.health_check)
        self.app.get("/")(self.root)
    
    def get_model_manager(self) -> TTSModelManager:
        """获取或创建模型管理器"""
        if self.model_manager is None:
            try:
                self.model_manager = TTSModelManager(self.config.model_repo)
                # 延迟加载模型，第一次使用时再加载
            except Exception as e:
                logger.error(f"Failed to create model manager: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to initialize model: {e}")
        return self.model_manager
    
    # ========== 管理端点实现 ==========
    
    async def get_speakers_list(self):
        """获取说话人列表（简化版本）"""
        speakers = scan_speakers(self.config.samples_dir)
        # 只返回ID列表
        speaker_ids = [s["id"] for s in speakers]
        return {"speakers": speaker_ids, "count": len(speaker_ids)}
    
    async def get_speakers(self):
        """获取说话人详细信息，返回SpeakerInfo对象数组"""
        speakers = scan_speakers(self.config.samples_dir)
        speaker_infos = []
        for s in speakers:
            # 将audio_path映射到file_path，如果audio_path为None则使用空字符串
            file_path = s["audio_path"] if s["audio_path"] else s["design_path"]
            speaker_infos.append(SpeakerInfo(
                name=s["id"],  # id映射到name
                type=s["type"],
                file_path=file_path,
                text_path=s["text_path"],
                design_path=s["design_path"],
                design_description=s["design_description"],
                voice_id=s["id"]  # voice_id使用原始id
            ))
        return speaker_infos
    
    async def get_languages(self):
        """获取支持的语言列表"""
        try:
            manager = self.get_model_manager()
            supported_langs = manager.get_supported_languages()
        except Exception as e:
            logger.warning(f"Failed to get languages from model manager, using default: {e}")
            supported_langs = SUPPORTED_LANGUAGES
        
        language_infos = []
        for code, name in supported_langs.items():
            language_infos.append(LanguageInfo(code=code, name=name))
        
        return LanguageListResponse(
            languages=language_infos,
            count=len(language_infos)
        )
    
    async def get_folders(self):
        """获取文件夹信息"""
        return FolderInfo(
            samples_dir=str(self.config.samples_dir),
            output_dir=str(self.config.output_dir)
        )
    
    async def get_models_list(self):
        """获取模型列表"""
        # 定义支持的模型
        models = [
            ModelInfo(
                name="OmniVoice",
                framework="omnivoice",
                repo="k2-fsa/OmniVoice"
            ),
            ModelInfo(
                name="LongCat-AudioDiT-1B",
                framework="longcat",
                repo="meituan-longcat/LongCat-AudioDiT-1B"
            ),
            ModelInfo(
                name="LongCat-AudioDiT-3.5B",
                framework="longcat",
                repo="meituan-longcat/LongCat-AudioDiT-3.5B"
            )
        ]
        
        # 获取当前模型
        current_model = None
        if self.model_manager:
            framework_info = self.model_manager.get_framework_info()
            for model in models:
                if model.repo == self.config.model_repo:
                    current_model = model
                    break
        
        return ModelListResponse(
            models=models,
            current_model=current_model
        )
    
    async def get_tts_settings(self):
        """获取TTS设置"""
        try:
            manager = self.get_model_manager()
            settings = manager.get_tts_settings()
        except Exception as e:
            logger.warning(f"Failed to get TTS settings, using default: {e}")
            settings = DEFAULT_TTS_SETTINGS
        
        return TTSSettingsInfo(**settings)
    
    async def get_sample(self, file_name: str):
        """获取样本音频文件"""
        file_path = Path(self.config.samples_dir) / file_name
        
        # 安全检查：确保文件在samples目录内
        try:
            file_path.resolve().relative_to(self.config.samples_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # 检查文件类型
        if file_path.suffix.lower() in ['.wav', '.mp3', '.flac']:
            return FileResponse(file_path, media_type="audio/wav")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # ========== 配置端点实现 ==========
    
    async def set_output_folder(self, request: OutputFolderRequest):
        """设置输出文件夹"""
        new_dir = Path(request.output_folder)
        try:
            new_dir.mkdir(parents=True, exist_ok=True)
            self.config.output_dir = new_dir
            return StatusResponse(
                status="ok",
                message=f"Output folder updated to {new_dir}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to set output folder: {e}")
    
    async def set_speaker_folder(self, request: SpeakerFolderRequest):
        """设置说话人文件夹"""
        new_dir = Path(request.speaker_folder)
        if not new_dir.exists():
            raise HTTPException(status_code=404, detail="Folder does not exist")
        
        self.config.samples_dir = new_dir
        return StatusResponse(
            status="ok",
            message=f"Speaker folder updated to {new_dir}"
        )
    
    async def switch_model(self, request: ModelNameRequest):
        """切换模型"""
        try:
            if self.model_manager:
                self.model_manager.reload_model(request.model_name)
            else:
                self.config.model_repo = request.model_name
                self.model_manager = TTSModelManager(request.model_name)
            
            return StatusResponse(
                status="ok",
                message=f"Model switched to {request.model_name}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to switch model: {e}")
    
    async def set_tts_settings(self, request: TTSSettingsRequest):
        """设置TTS参数"""
        try:
            manager = self.get_model_manager()
            settings_dict = request.dict()
            manager.update_tts_settings(settings_dict)
            
            return StatusResponse(
                status="ok",
                message="TTS settings updated"
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update TTS settings: {e}")
    
    # ========== TTS生成端点实现 ==========
    
    async def tts_stream(
        self,
        text: str = Query(..., description="要合成的文本"),
        speaker_wav: str = Query(..., description="说话人参考音频路径"),
        language: str = Query("en", description="语言代码")
    ):
        """流式TTS生成"""
        # Unsupported
        raise HTTPException(status_code=501, detail=f"LocalSotaTalk does not support streaming")
    
    async def tts_to_audio(self, request: SynthesisRequest):
        """TTS生成音频数据，speaker_wav参数被视为voice_id"""
        try:
            # 将speaker_wav参数作为voice_id处理，查找对应的说话人信息
            voice_id = request.speaker_wav
            speaker = get_speaker_by_id(self.config.samples_dir, voice_id)
            
            if not speaker:
                # 获取所有可用的说话人ID
                all_speakers = scan_speakers(self.config.samples_dir)
                available_ids = [s["id"] for s in all_speakers]
                raise HTTPException(
                    status_code=404, 
                    detail=f"Speaker with voice_id '{voice_id}' not found. Available speakers: {', '.join(available_ids)}"
                )
            
            manager = self.get_model_manager()
            audio_data = None
            
            # 根据说话人类型选择合成方法
            if speaker.get("audio_path"):
                # 有音频文件，使用语音克隆模式
                audio_data = manager.synthesize(
                    text=request.text,
                    speaker_wav=str(speaker["audio_path"]),  # 传递音频文件路径
                    language=request.language
                )
            elif speaker.get("design_description") or speaker.get("design_path"):
                # 有设计描述，使用音频设计模式
                design_description = speaker.get("design_description", "")
                if not design_description and speaker.get("design_path"):
                    # 从设计文件读取描述
                    try:
                        design_description = Path(speaker["design_path"]).read_text(encoding="utf-8").strip()
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to read design description from {speaker['design_path']}: {e}"
                        )
                
                # 使用instructive合成
                try:
                    audio_data = manager.synthesize_instructively(
                        text=request.text,
                        design_description=design_description,
                        language=request.language
                    )
                except NotImplementedError as e:
                    # 当前模型不支持音频设计合成
                    raise HTTPException(
                        status_code=400,
                        detail=f"Current model does not support voice design. {e}"
                    )
            else:
                # 既没有音频文件也没有设计描述
                raise HTTPException(
                    status_code=400,
                    detail=f"Speaker '{voice_id}' has neither audio file nor design description. "
                           f"Add either '{voice_id}.wav' or '{voice_id}.design.txt' to samples directory."
                )
            
            # 计算时长
            duration = len(audio_data) / 24000  # 假设24kHz
            
            # 将音频数据转换为字节流
            import io
            import soundfile as sf
            
            # 创建内存中的音频文件
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, 24000, format='WAV')
            audio_buffer.seek(0)
            
            # 返回音频流响应
            return StreamingResponse(
                audio_buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=tts_output.wav",
                    "Duration": str(duration),
                    "Sample-Rate": "24000"
                }
            )
            
        except HTTPException:
            # 重新抛出HTTPException，让FastAPI处理
            raise
        except Exception as e:
            logger.error(f"TTS to audio failed: {e}")
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")
    
    async def tts_to_file(self, request: SynthesisFileRequest):
        """TTS生成并保存到文件，speaker_wav参数被视为voice_id"""
        try:
            # 将speaker_wav参数作为voice_id处理，查找对应的说话人信息
            voice_id = request.speaker_wav
            speaker = get_speaker_by_id(self.config.samples_dir, voice_id)
            
            if not speaker:
                # 获取所有可用的说话人ID
                all_speakers = scan_speakers(self.config.samples_dir)
                available_ids = [s["id"] for s in all_speakers]
                raise HTTPException(
                    status_code=404, 
                    detail=f"Speaker with voice_id '{voice_id}' not found. Available speakers: {', '.join(available_ids)}"
                )
            
            manager = self.get_model_manager()
            audio_data = None
            
            # 根据说话人类型选择合成方法
            if speaker.get("audio_path"):
                # 有音频文件，使用语音克隆模式
                audio_data = manager.synthesize(
                    text=request.text,
                    speaker_wav=str(speaker["audio_path"]),  # 传递音频文件路径
                    language=request.language
                )
            elif speaker.get("design_description") or speaker.get("design_path"):
                # 有设计描述，使用音频设计模式
                design_description = speaker.get("design_description", "")
                if not design_description and speaker.get("design_path"):
                    # 从设计文件读取描述
                    try:
                        design_description = Path(speaker["design_path"]).read_text(encoding="utf-8").strip()
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to read design description from {speaker['design_path']}: {e}"
                        )
                
                # 使用instructive合成
                try:
                    audio_data = manager.synthesize_instructively(
                        text=request.text,
                        design_description=design_description,
                        language=request.language
                    )
                except NotImplementedError as e:
                    # 当前模型不支持音频设计合成
                    raise HTTPException(
                        status_code=400,
                        detail=f"Current model does not support voice design. {e}"
                    )
            else:
                # 既没有音频文件也没有设计描述
                raise HTTPException(
                    status_code=400,
                    detail=f"Speaker '{voice_id}' has neither audio file nor design description. "
                           f"Add either '{voice_id}.wav' or '{voice_id}.design.txt' to samples directory."
                )
            
            # 生成文件名
            if "/" in request.file_name_or_path or "\\" in request.file_name_or_path:
                # 如果是完整路径
                file_path = Path(request.file_name_or_path)
            else:
                # 如果是文件名，保存到输出目录
                file_path = self.config.output_dir / request.file_name_or_path
            
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存音频文件
            save_audio_file(audio_data, file_path.parent, file_path.name)
            
            # 计算时长
            duration = len(audio_data) / 24000
            
            return SynthesisResponse(
                audio_data=None,
                file_path=str(file_path),
                duration=duration,
                sample_rate=24000
            )
            
        except HTTPException:
            # 重新抛出HTTPException，让FastAPI处理
            raise
        except Exception as e:
            logger.error(f"TTS to file failed: {e}")
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")
    
    # ========== 辅助端点 ==========
    
    async def health_check(self):
        """健康检查"""
        status = "healthy"
        message = "Service is running"
        
        try:
            manager = self.get_model_manager()
            # 尝试获取模型信息但不强制加载
            _ = manager.get_framework_info()
        except Exception as e:
            status = "degraded"
            message = f"Model not loaded: {e}"
        
        return StatusResponse(
            status=status,
            message=message
        )
    
    async def root(self):
        """根端点"""
        return {
            "service": "TTS Backend",
            "version": "0.1.0",
            "frameworks": ["OmniVoice", "LongCat-AudioDiT"],
            "docs": "/docs",
            "openapi": "/openapi.json"
        }


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """创建FastAPI应用"""
    if config is None:
        config = get_default_config()
    
    api = TTSAPI(config)
    
    # 配置CORS中间件，允许所有跨域请求
    api.app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有HTTP方法
        allow_headers=["*"],  # 允许所有头部
    )
    
    return api.app
