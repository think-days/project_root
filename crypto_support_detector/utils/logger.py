"""utils/logger.py - 日志配置工具"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
        level: str = "INFO",
        log_file: Optional[str] = None,
        format_string: Optional[str] = None
) -> logging.Logger:
    """设置日志配置"""

    # 默认格式
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    # 创建根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # 清除现有处理器
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)

    # 设置第三方库的日志级别
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    return root_logger


class LoggerAdapter(logging.LoggerAdapter):
    """自定义日志适配器，添加额外的上下文信息"""

    def process(self, msg, kwargs):
        """处理日志消息"""
        # 添加时间戳
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["timestamp"] = datetime.now().isoformat()

        # 添加自定义上下文
        if hasattr(self, 'context'):
            kwargs["extra"].update(self.context)

        return msg, kwargs
