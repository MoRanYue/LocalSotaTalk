#!/usr/bin/env python3
"""TTS Backend Service Main Entry"""
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config, create_config_from_args, parse_args
from api.endpoints import create_app
import uvicorn


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Create FastAPI application
    app = create_app(config)
    
    # Print startup information
    print("=" * 60)
    print("TTS Backend Service Starting")
    print(f"Model: {config.model_repo}")
    print(f"Framework: {config.framework}")
    print(f"Samples Directory: {config.samples_dir}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Server: {config.host}:{config.port}")
    print("=" * 60)
    print(f"API Documentation: http://{config.host}:{config.port}/docs")
    print(f"OpenAPI Specification: http://{config.host}:{config.port}/openapi.json")
    print("=" * 60)
    
    # Start server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nService stopped")
        sys.exit(0)
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)