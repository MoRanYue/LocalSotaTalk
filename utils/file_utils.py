"""文件工具函数，用于扫描说话人样本和音频文件管理"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import soundfile as sf
import numpy as np
from .constants import SPEAKER_TYPES


def scan_speakers(samples_dir: Union[str, Path]) -> List[Dict]:
    """
    扫描说话人样本目录，返回说话人信息列表
    
    Args:
        samples_dir: 样本目录路径
        
    Returns:
        List[Dict]: 说话人信息列表，每个字典包含:
            - id: 说话人ID
            - type: 说话人类型 (audio_only, audio_with_text, design_only)
            - audio_path: 音频文件路径（如果有）
            - text_path: 文本文件路径（如果有）
            - design_path: 设计文件路径（如果有）
            - design_description: 设计描述（如果有）
    """
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        return []
    
    speakers = []
    speaker_map = {}
    
    # 第一轮扫描：处理所有.wav和.design.txt文件
    for file_path in samples_dir.iterdir():
        if file_path.suffix == ".wav":
            speaker_id = file_path.stem
            speaker_map[speaker_id] = speaker_map.get(speaker_id, {})
            speaker_map[speaker_id]["audio_path"] = file_path
            
            # 检查对应的.txt文件
            txt_file = file_path.with_suffix(".txt")
            if txt_file.exists():
                speaker_map[speaker_id]["text_path"] = txt_file
            
            # 检查对应的.design.txt文件
            design_file = file_path.with_suffix(".design.txt")
            if design_file.exists():
                speaker_map[speaker_id]["design_path"] = design_file
                try:
                    speaker_map[speaker_id]["design_description"] = design_file.read_text(encoding="utf-8").strip()
                except:
                    speaker_map[speaker_id]["design_description"] = ""
        
        elif file_path.suffix == ".txt" and file_path.stem.endswith(".design"):
            # .design.txt文件
            speaker_id = file_path.stem.replace(".design", "")
            speaker_map[speaker_id] = speaker_map.get(speaker_id, {})
            speaker_map[speaker_id]["design_path"] = file_path
            try:
                speaker_map[speaker_id]["design_description"] = file_path.read_text(encoding="utf-8").strip()
            except:
                speaker_map[speaker_id]["design_description"] = ""
    
    # 第二轮：构建说话人信息
    for speaker_id, info in speaker_map.items():
        speaker_type = "audio_only"
        
        if "audio_path" in info and "text_path" in info:
            speaker_type = "audio_with_text"
        elif "design_path" in info and "audio_path" not in info:
            speaker_type = "design_only"
        elif "design_path" in info and "audio_path" in info:
            # 既有音频又有设计文件，优先使用音频+文本类型
            if "text_path" in info:
                speaker_type = "audio_with_text"
            else:
                speaker_type = "audio_only"
        
        speaker_info = {
            "id": speaker_id,
            "type": speaker_type,
            "audio_path": str(info.get("audio_path")) if "audio_path" in info else None,
            "text_path": str(info.get("text_path")) if "text_path" in info else None,
            "design_path": str(info.get("design_path")) if "design_path" in info else None,
            "design_description": info.get("design_description")
        }
        speakers.append(speaker_info)
    
    return speakers


def get_speaker_by_id(samples_dir: Union[str, Path], speaker_id: str) -> Optional[Dict]:
    """
    根据说话人ID获取说话人信息
    
    Args:
        samples_dir: 样本目录路径
        speaker_id: 说话人ID
        
    Returns:
        Optional[Dict]: 说话人信息，如果不存在返回None
    """
    speakers = scan_speakers(samples_dir)
    for speaker in speakers:
        if speaker["id"] == speaker_id:
            return speaker
    return None


def save_audio_file(
    audio_data: np.ndarray,
    output_dir: Union[str, Path],
    filename: str,
    sample_rate: int = 24000
) -> str:
    """
    保存音频文件到输出目录
    
    Args:
        audio_data: 音频数据数组
        output_dir: 输出目录路径
        filename: 文件名（无需扩展名）
        sample_rate: 采样率
        
    Returns:
        str: 保存的文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 确保文件名以.wav结尾
    if not filename.lower().endswith(".wav"):
        filename = f"{filename}.wav"
    
    filepath = output_dir / filename
    
    # 保存音频文件
    sf.write(filepath, audio_data, sample_rate)
    
    return str(filepath)


def get_audio_duration(filepath: Union[str, Path]) -> float:
    """
    获取音频文件时长（秒）
    
    Args:
        filepath: 音频文件路径
        
    Returns:
        float: 音频时长（秒）
    """
    try:
        with sf.SoundFile(str(filepath)) as audio:
            return len(audio) / audio.samplerate
    except Exception:
        return 0.0


def read_audio_file(filepath: Union[str, Path], sample_rate: int = 24000) -> Optional[Tuple[np.ndarray, int]]:
    """
    读取音频文件
    
    Args:
        filepath: 音频文件路径
        sample_rate: 目标采样率
        
    Returns:
        Optional[Tuple[np.ndarray, int]]: (音频数据, 实际采样率)，读取失败返回None
    """
    try:
        data, sr = sf.read(str(filepath))
        # 如果是单声道，确保形状正确
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        # 如果需要重采样
        if sr != sample_rate:
            # 这里可以使用librosa进行重采样，但为了简化先不实现
            # 暂时假设采样率匹配
            pass
        return data, sr
    except Exception as e:
        print(f"Error reading audio file {filepath}: {e}")
        return None


def list_audio_files(directory: Union[str, Path]) -> List[str]:
    """
    列出目录中的所有音频文件
    
    Args:
        directory: 目录路径
        
    Returns:
        List[str]: 音频文件路径列表
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    audio_files = []
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        audio_files.extend([str(f) for f in directory.glob(f"*{ext}")])
    
    return sorted(audio_files)


def validate_audio_file(filepath: Union[str, Path]) -> bool:
    """
    验证音频文件是否有效
    
    Args:
        filepath: 音频文件路径
        
    Returns:
        bool: 是否有效
    """
    try:
        with sf.SoundFile(str(filepath)) as audio:
            # 检查文件是否可以正常打开和读取
            _ = audio.read(1)
            return True
    except Exception:
        return False