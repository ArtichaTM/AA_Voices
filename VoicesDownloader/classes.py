from typing import Optional
import asyncio
from enum import IntEnum, auto
from time import time
from json import loads, dump
from pathlib import Path

from aiohttp import ClientSession


__all__ = (
    'Ascension',
    'Downloader',
    'VoiceLine',
    'ServantVoices',
    'MAINDIR'
)


MAINDIR = Path() / 'VoicesDownloader' / 'downloads'


class Ascension(IntEnum):
    Asc0 = auto()
    Asc1 = auto()
    Asc2 = auto()
    Asc3 = auto()
    Asc4 = auto()


class Downloader:
    __slots__ = (
        'delay', 'timeout', 'timestamps', 'last_request', 'session'
    )
    timestamps: dict[str, float]
    _instance: Optional['Downloader'] = None
    API_SERVER: str = r'https://api.atlasacademy.io'

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, delay: float = 1, timeout: float = 10) -> None:
        self.delay = delay
        self.timeout = timeout
        self.last_request = time()
        self.session = ClientSession()

    async def updateInfo(self):
        info = await self.request('/info')
        self.timestamps = {region: data['timestamp'] for region, data in info.items() if region in {'NA', 'JP'}}

    async def request(self, address: str, params: dict = dict()) -> dict:
        assert isinstance(address, str)
        assert isinstance(params, dict)
        assert all([isinstance(key, str) for key in params.keys()])

        while time() - self.last_request > self.delay:
            await asyncio.sleep(time() - self.last_request)

        self.last_request = time() + self.timeout
        async with self.session.get(self.API_SERVER + address, params=params, allow_redirects=False) as response:
            self.last_request = time()
            return loads(await response.text())

    async def download(self, address: str, save_path: Path, params: dict | None = None) -> None:
        if params is None: params = dict()
        while (time() - self.last_request) > self.delay:
            await asyncio.sleep(time() - self.last_request)

        self.last_request = time() + self.timeout
        async with self.session.get(self.API_SERVER + address, allow_redirects=False, params=params) as response:
            self.last_request = time()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open('w+', encoding='utf-8') as f: 
                text = await response.text()
                print(f"Writing to {save_path.absolute()} text {text[:100]}")
                f.write(text)

    async def destroy(self) -> None:
        assert isinstance(self.session, ClientSession)
        await self.session.close()


class VoiceLine:
    pass


class ServantVoices:
    ANNOTATION_VOICES = dict[Ascension, dict[str, VoiceLine]]
    __slots__ = (
        'id', 'voices'
    )
    voices: ANNOTATION_VOICES
    SERVANTS_FOLDER = MAINDIR / 'Servants'

    def __init__(self, id: int, voices: ANNOTATION_VOICES):
        self.id = id
        self.voices = voices

    @classmethod
    async def updateJSON(cls, id: int) -> None:
        servant_folder = cls.SERVANTS_FOLDER / f"{id}"
        servant_json_path = servant_folder / 'info.json'
        servant_json_path.parent.mkdir(parents=True, exist_ok=True)
        servant_json_path.touch(exist_ok=True)

        await Downloader().download(
            address=f'/nice/NA/servant/{id}',
            save_path=servant_json_path,
            params={'lore': 'true'}
        )

    @classmethod
    async def load(cls, id: int) -> None:
        servant_folder = cls.SERVANTS_FOLDER / f"{id}"
        servant_json_path = servant_folder / 'info.json'

        while True:
            # JSON doesn't exist
            if not servant_json_path.exists():
                await cls.updateJSON(id=id)
            # Old JSON
            elif servant_json_path.lstat().st_mtime < Downloader().timestamps['NA']:
                await cls.updateJSON(id=id)
            break

        