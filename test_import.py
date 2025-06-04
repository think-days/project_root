# test_import.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试导入"""

try:
    print("Testing imports...")

    # 测试标准库
    import asyncio
    import logging

    print("✓ Standard libraries OK")

    # 测试第三方库
    import pandas as pd
    import numpy as np
    import ccxt

    print("✓ Third-party libraries OK")

    # 测试项目模块
    from crypto_support_detector.config.settings import config

    print("✓ Config module OK")

    from crypto_support_detector.data.fetcher import DataFetcher

    print("✓ Data module OK")

    from crypto_support_detector.monitoring.scanner import RealTimeScanner

    print("✓ Monitoring module OK")

    print("\nAll imports successful! You can run main.py")

except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease install missing dependencies:")
    print("pip install -r requirements.txt")
