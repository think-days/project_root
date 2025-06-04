# crypto_support_detector/__init__.py
"""Crypto Support Detector Package"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config.settings import config
from .monitoring.scanner import RealTimeScanner
from .data.fetcher import DataFetcher

__all__ = ['config', 'RealTimeScanner', 'DataFetcher']
