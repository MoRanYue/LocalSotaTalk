#!/usr/bin/env python3
"""测试utils模块"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("测试utils模块导入...")

# 测试utils导入
try:
    from utils import constants
    print("✅ 成功导入utils.constants")
    
    # 测试常量访问
    print(f"✅ 访问SUPPORTED_LANGUAGES: {len(constants.SUPPORTED_LANGUAGES)}种语言")
except ImportError as e:
    print(f"❌ 导入utils.constants失败: {e}")
    print(f"Python路径: {sys.path}")
    
    # 检查utils目录
    utils_dir = project_root / "utils"
    print(f"utils目录: {utils_dir}, 存在: {utils_dir.exists()}")
    print(f"utils/__init__.py: {utils_dir / '__init__.py'}, 存在: {(utils_dir / '__init__.py').exists()}")

try:
    from utils import file_utils
    print("✅ 成功导入utils.file_utils")
except ImportError as e:
    print(f"❌ 导入utils.file_utils失败: {e}")

# 测试从utils.constants直接导入
try:
    from utils.constants import SUPPORTED_LANGUAGES
    print(f"✅ 直接从utils.constants导入成功: {len(SUPPORTED_LANGUAGES)}种语言")
except ImportError as e:
    print(f"❌ 直接从utils.constants导入失败: {e}")

# 测试从utils.file_utils直接导入
try:
    from utils.file_utils import scan_speakers
    print("✅ 直接从utils.file_utils导入scan_speakers成功")
except ImportError as e:
    print(f"❌ 直接从utils.file_utils导入失败: {e}")