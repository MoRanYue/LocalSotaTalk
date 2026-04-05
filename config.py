import argparse
import os
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
            str: 框架类型 ("omnivoice" 或 "longcat")
        """
        repo_lower = repo.lower()
        
        # 检查LongCat-AudioDiT模式
        if any(keyword in repo_lower for keyword in ["longcat", "audiodit", "meituan"]):
            return "longcat"
        
        # 检查OmniVoice模式
        if any(keyword in repo_lower for keyword in ["omnivoice", "k2-fsa"]):
            return "omnivoice"
        
        # 默认使用OmniVoice，因为它支持更多语言
        return "omnivoice"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="TTS Backend Service - Supports LongCat-AudioDiT and OmniVoice"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="HuggingFace model repository (e.g., k2-fsa/OmniVoice or meituan-longcat/LongCat-AudioDiT-1B)"
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
        log_file=args.log_file
    )


def get_default_config() -> AppConfig:
    """Get default configuration"""
    return create_config_from_args(parse_args())
