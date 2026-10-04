"""In-memory LRU store for rendered live charts (served as thumbnails to the UI)."""

from __future__ import annotations

import secrets
import threading
from collections import OrderedDict


class ChartStore:
    def __init__(self, capacity: int = 200) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, png: bytes) -> str:
        chart_id = secrets.token_hex(8)
        with self._lock:
            self._items[chart_id] = png
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return chart_id

    def get(self, chart_id: str) -> bytes | None:
        with self._lock:
            png = self._items.get(chart_id)
            if png is not None:
                self._items.move_to_end(chart_id)
            return png

    def __len__(self) -> int:
        return len(self._items)
