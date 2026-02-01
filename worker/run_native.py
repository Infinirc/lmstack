#!/usr/bin/env python3
"""
Native Worker Agent launcher for macOS.

This script runs the worker agent directly without Docker,
suitable for Mac with Apple Silicon.

Usage:
    # First, get a registration token from LMStack web UI

    # Then run:
    python run_native.py --server-url http://YOUR_LMSTACK_SERVER:8000 --registration-token YOUR_TOKEN

    # Or set environment variables:
    export BACKEND_URL=http://YOUR_LMSTACK_SERVER:8000
    export REGISTRATION_TOKEN=YOUR_TOKEN
    python run_native.py
"""

import os
import sys

if __name__ == "__main__":
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Import here to avoid E402
    from agent import main

    print("=" * 60)
    print("LMStack Worker Agent (Native Mode)")
    print("=" * 60)
    print()
    print("This worker runs natively without Docker.")
    print("Supported backends: Ollama, MLX-LM, llama.cpp")
    print()
    print("Prerequisites:")
    print("  - Ollama: brew install ollama && ollama serve")
    print("  - MLX-LM: pip install mlx-lm")
    print("  - llama.cpp: brew install llama.cpp")
    print()
    print("=" * 60)
    print()

    main()
