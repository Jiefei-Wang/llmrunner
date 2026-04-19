from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        raise NotImplementedError
