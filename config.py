import argparse
import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class AppConfig:
    """Application configuration class"""
    model_repo: str
    samples_dir: Path
    output_dir: Path
    host: str
    port: int
    log_level: str
    log_file: Optional[Path] = None
    device: str = "auto"
    
    # Framework detection
    framework: str = field(init=False)
    
    def __post_init__(self):
        """Auto-detect framework after initialization"""
        self.samples_dir = Path(self.samples_dir)
        self.output_dir = Path(self.output_dir)
        if self.log_file:
            self.log_file = Path(self.log_file)
        
        # Detect framework type based on model repo name
        self.framework = self._detect_framework_from_repo(self.model_repo)
    
    def _detect_framework_from_repo(self, repo: str) -> str:
        """
        根据模型仓库名称检测框架类型
        
        Args:
            repo: 模型仓库名称或路径
            
        Returns:
            str: 框架类型 ("voxcpm", "omnivoice", "longcat" 或 "mosstts")
        """
        repo_lower = repo.lower()
        
        # 检查VoxCPM模式
        if any(keyword in repo_lower for keyword in ["voxcpm", "vox-cpm"]):
            return "voxcpm"
        
        # 检查LongCat-AudioDiT模式
        if any(keyword in repo_lower for keyword in ["longcat", "audiodit", "meituan"]):
            return "longcat"
        
        # 检查OmniVoice模式
        if any(keyword in repo_lower for keyword in ["omnivoice", "k2-fsa"]):
            return "omnivoice"
        
        # 检查MOSS-TTS模式（排除MOSS-TTS-Nano）
        if any(keyword in repo_lower for keyword in ["moss-tts", "mosstts", "moss"]) and "nano" not in repo_lower:
            return "mosstts"
        
        # 默认使用VoxCPM（目前最新/活跃的框架）
        return "voxcpm"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="TTS Backend Service - Supports VoxCPM, LongCat-AudioDiT and OmniVoice"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="HuggingFace model repository (e.g., k2-fsa/OmniVoice, meituan-longcat/LongCat-AudioDiT-1B, or a VoxCPM local path)"
    )
    
    # Directory configuration
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="./samples",
        help="Speaker samples directory path"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="./output",
        help="Output audio directory path"
    )
    
    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server bind address"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run the model on (auto, cpu, or cuda)"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="./logs/tts_server.log",
        help="Log file path"
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> AppConfig:
    """Create configuration object from command line arguments"""
    # Ensure directories exist
    os.makedirs(args.samples_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    return AppConfig(
        model_repo=args.model,
        samples_dir=args.samples_dir,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        log_file=args.log_file,
        device=args.device
    )


def get_default_config() -> AppConfig:
    """Get default configuration"""
    return create_config_from_args(parse_args())