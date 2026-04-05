"""API数据模型（基于openapi.json定义）"""
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, model_validator
import uuid
from datetime import datetime


# 错误处理
class ValidationError(BaseModel):
    """验证错误"""
    loc: List[str]
    msg: str
    type: str
    input: Optional[Any] = None
    ctx: Optional[Dict[str, Any]] = None


class HTTPValidationError(BaseModel):
    """HTTP验证错误"""
    detail: List[ValidationError]


# 请求模型
class ModelNameRequest(BaseModel):
    """模型名称请求"""
    model_name: str = Field(..., description="模型名称")


class OutputFolderRequest(BaseModel):
    """输出文件夹请求"""
    output_folder: str = Field(..., description="输出文件夹路径")


class SpeakerFolderRequest(BaseModel):
    """说话人文件夹请求"""
    speaker_folder: str = Field(..., description="说话人文件夹路径")


class SynthesisRequest(BaseModel):
    """语音合成请求"""
    text: str = Field(..., description="要合成的文本")
    speaker_wav: str = Field(..., description="说话人参考音频路径")
    language: str = Field(default="en", description="语言代码")


class SynthesisFileRequest(BaseModel):
    """语音合成到文件请求"""
    text: str = Field(..., description="要合成的文本")
    speaker_wav: str = Field(..., description="说话人参考音频路径")
    language: str = Field(default="en", description="语言代码")
    file_name_or_path: str = Field(..., description="文件名或路径")


class TTSSettingsRequest(BaseModel):
    """TTS设置请求"""
    stream_chunk_size: int = Field(default=1024, description="流式块大小")
    temperature: float = Field(default=1.0, ge=0.0, le=5.0, description="温度")
    speed: float = Field(default=1.0, gt=0.0, le=5.0, description="语速")
    length_penalty: float = Field(default=1.0, description="长度惩罚")
    repetition_penalty: float = Field(default=1.0, description="重复惩罚")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p采样")
    top_k: int = Field(default=50, ge=1, description="Top-k采样")
    enable_text_splitting: bool = Field(default=True, description="启用文本分割")


# 响应模型
class SpeakerInfo(BaseModel):
    """说话人信息"""
    name: str = Field(..., description="说话人名称")
    type: str = Field(..., description="说话人类型 (audio_only, audio_with_text, design_only)")
    file_path: str = Field(..., description="音频文件路径")
    text_path: Optional[str] = Field(None, description="文本文件路径")
    design_path: Optional[str] = Field(None, description="设计文件路径")
    design_description: Optional[str] = Field(None, description="设计描述")
    voice_id: Optional[str] = Field(None, description="语音ID，与name相同")
    
    @model_validator(mode='after')
    def set_voice_id(self):
        """设置voice_id，使其与name相同"""
        if self.voice_id is None:
            self.voice_id = self.name
        return self


class LanguageInfo(BaseModel):
    """语言信息"""
    code: str = Field(..., description="语言代码")
    name: str = Field(..., description="语言名称")


class ModelInfo(BaseModel):
    """模型信息"""
    name: str = Field(..., description="模型名称")
    framework: str = Field(..., description="框架类型 (omnivoice, longcat)")
    repo: str = Field(..., description="HuggingFace仓库")


class FolderInfo(BaseModel):
    """文件夹信息"""
    samples_dir: str = Field(..., description="样本目录路径")
    output_dir: str = Field(..., description="输出目录路径")


class TTSSettingsInfo(BaseModel):
    """TTS设置信息"""
    stream_chunk_size: int = Field(..., description="流式块大小")
    temperature: float = Field(..., description="温度")
    speed: float = Field(..., description="语速")
    length_penalty: float = Field(..., description="长度惩罚")
    repetition_penalty: float = Field(..., description="重复惩罚")
    top_p: float = Field(..., description="Top-p采样")
    top_k: int = Field(..., description="Top-k采样")
    enable_text_splitting: bool = Field(..., description="启用文本分割")


class SynthesisResponse(BaseModel):
    """语音合成响应"""
    audio_data: Optional[List[float]] = Field(None, description="音频数据（如果返回数据）")
    file_path: Optional[str] = Field(None, description="文件路径（如果保存到文件）")
    duration: float = Field(..., description="音频时长（秒）")
    sample_rate: int = Field(default=24000, description="采样率")


class StreamChunk(BaseModel):
    """流式数据块"""
    chunk_id: int = Field(..., description="块ID")
    data: List[float] = Field(..., description="音频数据")
    is_final: bool = Field(default=False, description="是否为最后一块")


class StatusResponse(BaseModel):
    """状态响应"""
    status: str = Field(default="ok", description="状态")
    message: Optional[str] = Field(None, description="消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


# 扩展模型（不在openapi.json中，但API需要）
class SpeakerListResponse(BaseModel):
    """说话人列表响应"""
    speakers: List[SpeakerInfo] = Field(..., description="说话人列表")
    count: int = Field(..., description="说话人数量")


class LanguageListResponse(BaseModel):
    """语言列表响应"""
    languages: List[LanguageInfo] = Field(..., description="语言列表")
    count: int = Field(..., description="语言数量")


class ModelListResponse(BaseModel):
    """模型列表响应"""
    models: List[ModelInfo] = Field(..., description="模型列表")
    current_model: Optional[ModelInfo] = Field(None, description="当前模型")


# 工具函数
def create_error_response(message: str, error_type: str = "validation_error") -> Dict[str, Any]:
    """创建错误响应"""
    return {
        "detail": [{
            "loc": ["body"],
            "msg": message,
            "type": error_type
        }]
    }