"""setup.py - 安装配置"""
from setuptools import setup, find_packages

setup(
    name="crypto-support-detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ccxt>=4.1.22",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "aiohttp>=3.9.1",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="Real-time cryptocurrency support level detector",
)
