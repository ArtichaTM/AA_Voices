import asyncio
from types import TracebackType
from time import time

from aiohttp import ClientSession, client_exceptions
from aiopath import AsyncPath

__all__ = ('Settings', 'Colors')


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class GetRequest:
    __slots__ = ('request', 'url')
    latest_request: float = 0

    def __init__(
        self,
        url: str
    ) -> None:
        assert Settings.i is not None
        if url.startswith('/'):
            url = Settings.i.api_url + url
        self.url = url

    async def __aenter__(self):
        assert Settings.i is not None
        while True:
            current_time = time()
            delay = current_time - self.latest_request - Settings.i.requests_delay
            # Delay >  0: There's some time left to wait
            # Delay <= 0: Delay already passed without requests
            if delay >= 0:
                break
            await asyncio.sleep(abs(delay))
        self.__class__.latest_request = time()
        times_requested = 0
        while True:
            times_requested += 1
            self.update_request()
            try:
                response = await self.request.__aenter__()
            except client_exceptions.ClientOSError as e:
                pass
            else:
                break
            if times_requested > Settings.i.maximum_retries:
                raise e
            await asyncio.sleep(times_requested**2)
        return response

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None
    ) -> None:
        await self.request.__aexit__(exc_type, exc, tb)

    def update_request(self) -> None:
        assert Settings.i is not None
        self.request = Settings.i.session.get(self.url, allow_redirects=False)


class Settings(dict):
    __slots__ = (
        'main_dir',
        'logs_path',
        'dataset_dir',
        'servants_path',
        'session',
        'album_name',
        'voices_folder_name',
        'api_url',
        'ffmpeg_path',
        'latest_aiohttp_request',
        'requests_delay',
        'replace_spaces',
        'maximum_retries'
    )
    i: 'Settings | None' = None
    main_dir: AsyncPath
    logs_path: AsyncPath
    dataset_dir: AsyncPath
    servants_path: AsyncPath
    session: ClientSession
    album_name: str
    voices_folder_name: str
    api_url: str
    ffmpeg_path: str
    requests_delay: float
    replace_spaces: bool
    maximum_retries: int

    def __init__(self):
        super().__init__()
        self.__class__.i = self

        # Paths
        self.main_dir = AsyncPath()
        self.logs_path = self.main_dir / 'logs'
        self.dataset_dir = self.main_dir / 'dataset'
        self.servants_path = self.dataset_dir / 'servants'

        # Strings
        self.album_name = 'Fate: Grand Order Servants'
        self.api_url = r'https://api.atlasacademy.io'
        self.voices_folder_name = 'voices'
        self.ffmpeg_path = 'ffmpeg'

        # Extra
        self.requests_delay = 1.0
        self.maximum_retries = 10
        self.replace_spaces = True

    async def __aenter__(self) -> None:
        await self.init()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def init(self) -> None:
        self.session = ClientSession()
        await asyncio.gather(
            self.logs_path.mkdir(exist_ok=True),
            self.dataset_dir.mkdir(exist_ok=True),
            self.servants_path.mkdir(exist_ok=True)
        )

    async def close(self) -> None:
        await self.session.close()

    def get(self, string: str) -> GetRequest:
        return GetRequest(string)
