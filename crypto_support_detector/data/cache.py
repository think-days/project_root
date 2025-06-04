"""data/cache.py - 数据缓存模块"""
import asyncio
import pickle
import time
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataCache:
    """简单的内存缓存实现"""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry['expires_at'] > time.time():
                    return entry['data']
                else:
                    # 过期，删除
                    del self._cache[key]
            return None

    async def set(self, key: str, data: Any, ttl: int = 300):
        """设置缓存数据"""
        async with self._lock:
            self._cache[key] = {
                'data': data,
                'expires_at': time.time() + ttl
            }

    async def delete(self, key: str):
        """删除缓存数据"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self):
        """清空缓存"""
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self):
        """清理过期缓存"""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry['expires_at'] <= current_time
            ]
            for key in expired_keys:
                del self._cache[key]
