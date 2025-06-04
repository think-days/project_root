"""monitoring/alerts.py - 告警系统"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from ..config.settings import config

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """告警类型"""
    APPROACHING_SUPPORT = "approaching_support"
    SUPPORT_BREAK = "support_break"
    STRONG_SUPPORT_FOUND = "strong_support_found"
    VOLUME_SPIKE = "volume_spike"
    MULTIPLE_TIMEFRAME_CONFLUENCE = "confluence"


@dataclass
class Alert:
    """告警数据结构"""
    alert_id: str
    alert_type: AlertType
    symbol: str
    timeframe: str
    level: float
    current_price: float
    message: str
    metadata: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'level': self.level,
            'current_price': self.current_price,
            'message': self.message,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.alert_handlers = []
        self.alert_history = defaultdict(list)
        self.cooldowns = defaultdict(lambda: datetime.min)
        self.cooldown_period = config.get('monitoring.alert_cooldown', 3600)
        self._running = False
        self._alert_queue = asyncio.Queue()
        self._process_task = None

    async def start(self):
        """启动告警管理器"""
        self._running = True
        self._process_task = asyncio.create_task(self._process_alerts())
        logger.info("Alert manager started")

    async def stop(self):
        """停止告警管理器"""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert manager stopped")

    def add_handler(self, handler):
        """添加告警处理器"""
        self.alert_handlers.append(handler)

    async def send_alert(self, **kwargs):
        """发送告警"""
        # 检查冷却时间
        key = f"{kwargs.get('symbol')}:{kwargs.get('alert_type')}:{kwargs.get('level', 0):.6f}"

        if datetime.now() < self.cooldowns[key]:
            logger.debug(f"Alert {key} is in cooldown period")
            return

        # 创建告警
        alert = Alert(
            alert_id=f"{key}:{datetime.now().timestamp()}",
            alert_type=AlertType(kwargs.get('alert_type', 'APPROACHING_SUPPORT')),
            symbol=kwargs.get('symbol'),
            timeframe=kwargs.get('timeframe'),
            level=kwargs.get('level', {}).level if hasattr(kwargs.get('level'), 'level') else kwargs.get('level', 0),
            current_price=kwargs.get('current_price', 0),
            message=self._generate_message(**kwargs),
            metadata=kwargs,
            timestamp=datetime.now()
        )

        # 更新冷却时间
        self.cooldowns[key] = datetime.now() + timedelta(seconds=self.cooldown_period)

        # 加入队列
        await self._alert_queue.put(alert)

    async def _process_alerts(self):
        """处理告警队列"""
        while self._running:
            try:
                # 等待告警
                alert = await asyncio.wait_for(
                    self._alert_queue.get(),
                    timeout=1.0
                )

                # 记录历史
                self.alert_history[alert.symbol].append(alert)

                # 限制历史记录数量
                if len(self.alert_history[alert.symbol]) > 100:
                    self.alert_history[alert.symbol] = self.alert_history[alert.symbol][-100:]

                # 调用所有处理器
                for handler in self.alert_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(alert)
                        else:
                            handler(alert)
                    except Exception as e:
                        logger.error(f"Error in alert handler: {e}")

                # 记录告警
                logger.info(f"Alert sent: {alert.message}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")

    def _generate_message(self, **kwargs) -> str:
        """生成告警消息"""
        alert_type = kwargs.get('alert_type')
        symbol = kwargs.get('symbol')
        timeframe = kwargs.get('timeframe')
        level = kwargs.get('level')
        current_price = kwargs.get('current_price', 0)

        if hasattr(level, 'level'):
            level_price = level.level
            level_score = level.score
            level_category = level.category
        else:
            level_price = level
            level_score = 0
            level_category = "UNKNOWN"

        if alert_type == "APPROACHING_SUPPORT":
            distance_pct = kwargs.get('distance_pct', 0) * 100
            return (
                f"📊 {symbol} ({timeframe}) approaching {level_category} support\n"
                f"Support: {level_price:.6f} (Score: {level_score:.3f})\n"
                f"Current: {current_price:.6f} ({distance_pct:.2f}% above)"
            )
        elif alert_type == "SUPPORT_BREAK":
            distance_pct = abs(kwargs.get('distance_pct', 0)) * 100
            return (
                f"⚠️ {symbol} ({timeframe}) breaking {level_category} support\n"
                f"Support: {level_price:.6f} (Score: {level_score:.3f})\n"
                f"Current: {current_price:.6f} ({distance_pct:.2f}% below)"
            )
        elif alert_type == "STRONG_SUPPORT_FOUND":
            return (
                f"💪 New STRONG support found for {symbol} ({timeframe})\n"
                f"Level: {level_price:.6f} (Score: {level_score:.3f})\n"
                f"Current: {current_price:.6f}"
            )
        else:
            return f"Alert: {alert_type} for {symbol} at {current_price:.6f}"

    def get_recent_alerts(self, symbol: Optional[str] = None,
                          hours: int = 24) -> List[Alert]:
        """获取最近的告警"""
        cutoff = datetime.now() - timedelta(hours=hours)

        if symbol:
            alerts = self.alert_history.get(symbol, [])
        else:
            alerts = []
            for symbol_alerts in self.alert_history.values():
                alerts.extend(symbol_alerts)

        return [a for a in alerts if a.timestamp > cutoff]
