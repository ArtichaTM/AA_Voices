import asyncio
from abc import ABC, abstractmethod

from progress.bar import Bar

class ProgressWatcher(ABC):
    __slots__ = ()

    @abstractmethod
    async def set_current(self, value: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def current_add(self, amount: int = 1) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def set_maximum(self, value: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def set_text(self, value: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def finish(self) -> None:
        raise NotImplementedError()


class NullProgressWatcher(ProgressWatcher):
    async def set_current(self, value: int) -> None:
        pass

    async def current_add(self, amount: int = 1) -> None:
        pass

    async def set_maximum(self, value: int) -> None:
        pass

    async def set_text(self, value: str) -> None:
        pass

    async def finish(self) -> None:
        pass


class PrintWatcher(ProgressWatcher):
    __slots__ = ('maximum', )
    async def set_current(self, value: int) -> None:
        print(f"current = {value}")

    async def current_add(self, amount: int = 1) -> None:
        print(f"current += {amount}")

    async def set_maximum(self, value: int) -> None:
        self.maximum = value
        print(f"Maximum={self.maximum}")

    async def set_text(self, value: str) -> None:
        print(f"text = {value}")

    async def finish(self) -> None:
        print("Finished!")


class ProgressBar(ProgressWatcher):
    __slots__ = ('bar', 'message_max')

    def __init__(self, message_max: int = 30) -> None:
        super().__init__()
        self.bar: Bar = Bar()
        self.bar.max = 0
        self.message_max = message_max
        self.update_suffix()

    def update_suffix(self) -> None:
        self.bar.suffix = f"{self.bar.index: <4}/{self.bar.max: <4}"

    def format_message(self, value: str) -> str:
        assert isinstance(value, str)
        value = value.replace('\n', ' ').replace('\r', '')
        if len(value) > self.message_max:
            value = value[:self.message_max]
        else:
            value = value.ljust(self.message_max)
        return value

    async def set_current(self, value: int) -> None:
        assert isinstance(value, int)
        self.bar.index = value
        self.update_suffix()
        self.bar.update()

    async def current_add(self, amount: int = 1) -> None:
        assert isinstance(amount, int)
        self.update_suffix()
        self.bar.next(amount)

    async def set_maximum(self, value: int) -> None:
        assert isinstance(value, int)
        self.update_suffix()
        self.bar.max = value

    async def set_text(self, value: str) -> None:
        assert isinstance(value, str)
        self.bar.message = self.format_message(value)
        self.bar.update()

    async def finish(self) -> None:
        self.bar.finish()


class ThrottleLastProgressBar(ProgressBar):
    __slots__ = ('bar', 'updater', 'throttle_time', 'message_max')

    def __init__(self, message_max: int = 30, throttle_time: int = 1) -> None:
        super().__init__()
        self.bar: Bar = Bar()
        self.throttle_time = throttle_time
        self.updater: asyncio.Task = None
        self.bar.max = 0
        self.message_max = message_max

    async def _update(self) -> None:
        try:
            await asyncio.sleep(self.throttle_time)
        except asyncio.CancelledError:
            pass
        else:
            self.update_suffix()
            self.bar.update()
        self.updater = None

    async def _start_update(self) -> None:
        if self.updater is not None:
            return
        self.updater = asyncio.create_task(self._update())

    async def set_current(self, value: int) -> None:
        self.bar.index = value
        await self._start_update()

    async def current_add(self, amount: int = 1) -> None:
        self.bar.index += amount
        await self._start_update()

    async def set_maximum(self, value: int) -> None:
        self.bar.max = value
        await self._start_update()

    async def set_text(self, value: str) -> None:
        self.bar.message = self.format_message(value)
        await self._start_update()

    async def finish(self) -> None:
        if self.updater is not None:
            self.updater.cancel()
        self.update_suffix()
        self.bar.update()
        await super().finish()
