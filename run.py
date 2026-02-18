#!/usr/bin/env python3
"""
Maya Voice AI - Main Entry Point

Usage:
    python run.py              # Start server on port 8000
    python run.py --port 8080  # Start server on custom port
"""

import os
import sys

# Set cuDNN 9 library path for faster-whisper (CTranslate2)
# This MUST be done before importing torch or any CUDA libraries
_cudnn_path = "/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/nvidia/cudnn/lib"
if os.path.exists(_cudnn_path):
    _ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _cudnn_path not in _ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_cudnn_path}:{_ld_path}"

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Maya - The World's Best Conversational Voice AI"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    print()
    print("=" * 60)
    print("  MAYA - The World's Best Conversational Voice AI")
    print("=" * 60)
    print()
    print("  Starting server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print()
    print("  Open in browser: http://localhost:{}/".format(args.port))
    print()
    print("=" * 60)
    print()

    # Validate config before starting
    from maya.config import validate_config
    if not validate_config():
        logger.error("Configuration validation failed!")
        sys.exit(1)

    # Start server
    from maya.server import run_server
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
