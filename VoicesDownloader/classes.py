from typing import Any, Generator, Optional
import asyncio
import subprocess
from enum import IntEnum
from time import time
from json import loads
from pathlib import Path
from functools import cached_property

from aiohttp import ClientSession


__all__ = (
    'Ascension',
    'Downloader',
    'VoiceLine',
    'ServantVoices',
    'MAINDIR'
)


class Ascension(IntEnum):
    Asc0 = 0
    Asc1 = 1
    Asc2 = 2
    Asc3 = 3
    Asc4 = 4
    Costume1 = 5
    Costume2 = 6
    Costume3 = 7
    Costume4 = 8
    Costume5 = 9
    Costume6 = 10
    Costume7 = 11
    Costume8 = 12


class VoiceLineCategory(IntEnum):
    Home = 0
    Growth = 1
    FirstGet = 2
    Batlle = 3
    TreasureDevice = 4
    EventReward = 5

    @classmethod
    def fromString(cls, value: str) -> 'VoiceLineCategory':
        match value:
            case 'home':
                return cls.Home
            case 'groeth':
                return cls.Growth
            case 'firstGet':
                return cls.FirstGet
            case 'battle':
                return cls.Batlle
            case 'treasureDevice':
                return cls.TreasureDevice
            case 'eventReward':
                return cls.EventReward
            case _:
                raise Exception(f"There's no such category: \"{value}\"")


MAINDIR = Path() / 'VoicesDownloader' / 'downloads'


class Downloader:
    __slots__ = (
        'delay', 'timeout', 'timestamps', 'last_request', 'session'
    )
    timestamps: dict[str, float]
    _instance: Optional['Downloader'] = None
    API_SERVER: str = r'https://api.atlasacademy.io'
    SERVANTS_FOLDER = MAINDIR / 'Servants'
    VOICES_FOLDER_NAME = 'voices'
    FFMPEG_PATH: str = 'ffmpeg'

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, delay: float = 1, timeout: float = 10) -> None:
        if hasattr(self, 'delay'): return
        self.delay = delay
        self.timeout = timeout
        self.last_request = time()
        self.session = ClientSession()

    async def updateInfo(self):
        info = await self.request('/info')
        self.timestamps = {region: data['timestamp'] for region, data in info.items() if region in {'NA', 'JP'}}

    async def request(self, address: str, params: dict = dict()) -> dict:
        assert isinstance(self.session, ClientSession)
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
        assert isinstance(self.session, ClientSession)
        assert isinstance(address, str)
        assert isinstance(save_path, Path)
        assert isinstance(params, dict) or params is None

        if params is None: params = dict()
        while (time() - self.last_request) > self.delay:
            await asyncio.sleep(time() - self.last_request)

        self.last_request = time() + self.timeout
        if address.startswith('https'):
            url = address
        else:
            url = self.API_SERVER + address
        async with self.session.get(url, allow_redirects=False, params=params) as response:
            self.last_request = time()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open('wb+') as f: 
                f.write(await response.read())

    async def destroy(self) -> None:
        assert isinstance(self.session, ClientSession)
        await self.session.close()
        self.session = None


class VoiceLine:
    __slots__ = (
        '__dict__', 'dictionary'
    )
    dictionary: dict[str, Any]
    loaded: bool

    def __init__(self, values: dict[str, Any]) -> None:
        assert isinstance(values, dict)
        assert 'svt_id' in values, \
            "Servant id (svt_id) should be added in dictionary passed to VoiceLine"
        self.dictionary = values
        self.loaded = False

    def __repr__(self) -> str:
        return f"<VL {self.name} for {self.ascension}>"

    @property
    def servant_id(self) -> int:
        return self.dictionary['svt_id']

    @cached_property
    def ascension(self) -> Ascension:
        return list(Ascension)[int(self.dictionary['id'][0][0])]

    @cached_property
    def type(self) -> VoiceLineCategory:
        return VoiceLineCategory.fromString(self.dictionary['svtVoiceType'])

    @property
    def name(self) -> str:
        if self.dictionary['overwriteName']:
            return self.dictionary['overwriteName']
        return self.dictionary['name']

    @property
    def overwriteName(self) -> str:
        return self.dictionary['overwriteName']

    @property
    def anyName(self) -> str:
        return self.overwriteName if self.overwriteName else self.name

    @property
    def downloaded(self) -> bool:
        return self.path.exists()

    @property
    def path_folder(self) -> Path:
        return Downloader.SERVANTS_FOLDER / str(self.servant_id) / Downloader.VOICES_FOLDER_NAME / \
            self.ascension.name / self.type.name / self.name

    @property
    def filename(self) -> str:
        return f"{self.anyName}.mp3"

    @property
    def path(self) -> Path:
        return self.path_folder / self.filename

    def voiceLinesURL(self) -> Generator[str, None, None]:
        yield from self.dictionary['audioAssets']

    @staticmethod
    def concat_mp3(
        source_paths: list[Path],
        target_path: Path,
        delete_source: bool = True,
        target_exist_ok: bool = True
    ) -> None:
        assert len({str(i.parent) for i in source_paths}) == 1  # Same parent folder

        if target_path.exists():
            if target_exist_ok:
                target_path.unlink()
            else:
                raise FileNotFoundError("Target existing when parameter target_exist_ok is False")

        filenames = [i.name for i in source_paths]
        if target_path.parent == source_paths[0].parent:
            target_str: str = target_path.name
        else:
            target_str: str = str(target_path.absolute())

        command = (
            f"{Downloader.FFMPEG_PATH} -i \"concat:"
            f"{'|'.join(filenames)}"
            '" -c copy '
            f'"{target_str}"'
        )
        subprocess.call(
            args=command,
            cwd=target_path.parent,
            timeout=2,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if delete_source:
            for source in source_paths:
                source.unlink()


    async def download(self) -> None:
        downloader = Downloader()
        paths: list[Path] = []
        for index, voice_url in enumerate(self.voiceLinesURL()):
            paths.append(self.path_folder / f"{self.anyName}_{index}.mp3")
            await downloader.download(voice_url, paths[-1])
        self.concat_mp3(paths, self.path, delete_source=True)


ANNOTATION_VOICE_CATEGORY = dict[VoiceLineCategory, dict[str, list[VoiceLine]]]
ANNOTATION_VOICES = dict[Ascension, ANNOTATION_VOICE_CATEGORY]
class ServantVoices:
    __slots__ = (
        'id', 'voice_lines', 'path'
    )
    id: int
    voice_lines: ANNOTATION_VOICES
    path: Path

    def __init__(self, id: int):
        self.id = id
        self.voice_lines = dict()
        self.path = Downloader.SERVANTS_FOLDER / str(id)

    def __repr__(self) -> str:
        return f"<Svt {self.id}" + (' with voice lines' if self.voice_lines else '') + '>'

    @classmethod
    async def _updateJSON(cls, id: int) -> None:
        servant_folder = Downloader.SERVANTS_FOLDER / f"{id}"
        servant_json_path = servant_folder / 'info.json'
        servant_json_path.parent.mkdir(parents=True, exist_ok=True)
        servant_json_path.touch(exist_ok=True)

        await Downloader().download(
            address=f'/nice/NA/servant/{id}',
            save_path=servant_json_path,
            params={'lore': 'true'}
        )

    @classmethod
    async def load(cls, id: int) -> 'ServantVoices':
        servant_folder = Downloader.SERVANTS_FOLDER / f"{id}"
        servant_json_path = servant_folder / 'info.json'

        while True:
            # JSON doesn't exist
            if not servant_json_path.exists():
                await cls._updateJSON(id=id)
            # Old JSON
            elif servant_json_path.lstat().st_mtime < Downloader().timestamps['NA']:
                await cls._updateJSON(id=id)
            break

        return ServantVoices(id=id)

    async def buildVoiceLinesDict(self, fill_all_ascensions: bool = False) -> None:
        data: dict = loads((self.path / 'info.json').read_text(encoding='utf-8'))
        output: ANNOTATION_VOICES = dict()
        for voices in data['profile']['voices']:
            type = VoiceLineCategory.fromString(voices['type'])
            ascension = list(Ascension)[voices['voicePrefix']]
            if ascension not in output:
                output[ascension] = dict()
            if type not in output[ascension]:
                output[ascension][type] = dict()
            for line in voices['voiceLines']:
                name = line['name']
                if name not in output[ascension][type]:
                    output[ascension][type][name] = []
                line['svt_id'] = self.id
                output[ascension][type][name].append(VoiceLine(line))

        if fill_all_ascensions:
            ascensionAdd = data['ascensionAdd']['voicePrefix']
            for ascension_str, target_ascension in ascensionAdd['ascension'].items():
                ascension_str: Ascension = list(Ascension)[int(ascension_str)]
                target_ascension: Ascension = list(Ascension)[target_ascension]
                assert target_ascension in output
                output[ascension_str] = output[target_ascension]

        self.voice_lines = output

    async def updateVoices(self) -> None:
        folder_voices = Downloader.SERVANTS_FOLDER / str(self.id) / Downloader.VOICES_FOLDER_NAME
        folder_voices.mkdir(exist_ok=True)
        for ascension, ascension_values in self.voice_lines.items():
            folder_ascension = folder_voices / ascension.name
            folder_ascension.mkdir(exist_ok=True)
            for category, category_values in ascension_values.items():
                folder_category = folder_ascension / category.name
                folder_category.mkdir(exist_ok=True)
                for type, type_values in category_values.items():
                    folder_type = folder_category / type
                    folder_type.mkdir(exist_ok=True)
                    for voice_line in type_values:
                        await voice_line.download()
