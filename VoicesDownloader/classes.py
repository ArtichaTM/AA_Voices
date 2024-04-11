from typing import Any, Generator, Optional
import asyncio
import atexit
import os
from logging import getLogger
import subprocess
from enum import IntEnum
from time import time
from json import loads
from pathlib import Path
from functools import cached_property

import aiohttp
from aiohttp import ClientSession
import aiohttp.client_exceptions
from progress.bar import Bar

__all__ = (
    'NoVoiceLines',
    'Ascension',
    'Downloader',
    'VoiceLine',
    'ServantVoices',
    'BasicServant',
    'MAINDIR'
)

logger = getLogger('AA_voices_downloader')

class NoVoiceLines(Exception): pass
class FFMPEGException(Exception): pass

"""
Analog to code below:

class Ascension(IntEnum):
    Asc0 = 0
    Asc1 = 1
    Asc2 = 2
    Asc3 = 3
    Asc4 = 4
    Costume0 = 5
    Costume1 = 6
    ...
    Costume19 = 24
"""
Ascension = IntEnum('Ascension', dict(
    list({
        f"Asc{i}": i for i in range(5)
    }.items())
    +
    list({
        f"Costume{i}": i+5 for i in range(20)
    }.items())
))


class VoiceLineCategory(IntEnum):
    Home = 0
    Growth = 1
    FirstGet = 2
    Battle = 3
    TreasureDevice = 4
    EventReward = 5
    MasterMission = 6
    EventShop = 7
    BoxGachaTalk = 8
    EventJoin = 9
    Guide = 10

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
                return cls.Battle
            case 'treasureDevice':
                return cls.TreasureDevice
            case 'eventReward':
                return cls.EventReward
            case 'masterMission':
                return cls.MasterMission
            case 'eventShop':
                return cls.EventShop
            case 'boxGachaTalk':
                return cls.BoxGachaTalk
            case 'eventJoin':
                return cls.EventJoin
            case 'guide':
                return cls.Guide
            case _:
                raise Exception(f"There's no such category: \"{value}\"")


MAINDIR = Path() / 'VoicesDownloader' / 'downloads'


class Downloader:
    __slots__ = (
        'delay', 'timeout', 'timestamps', 'last_request', 'session',
        'basic_servant'
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
        assert isinstance(delay, (int, float))
        assert isinstance(timeout, (int, float))
        if hasattr(self, 'delay'): return
        self.delay = delay
        self.timeout = timeout
        self.last_request = time()
        self.session = ClientSession()
        atexit.register(self.destroy)

    async def updateInfo(self) -> None:
        info = await self.request('/info')
        assert isinstance(info, dict)
        self.timestamps = {region: data['timestamp'] for region, data in info.items() if region in {'NA', 'JP'}}
        assert len(self.timestamps) > 0

    async def updateBasicServant(self) -> None:
        json = await self.request('/export/NA/basic_servant.json')
        assert isinstance(json, list)
        self.basic_servant: BasicServant = BasicServant(json)

    async def request(self, address: str, params: dict = dict()) -> dict | list:
        assert isinstance(self.session, ClientSession)
        assert isinstance(address, str)
        assert isinstance(params, dict)
        assert all([isinstance(key, str) for key in params.keys()])
        assert all([isinstance(key, (str, int, float)) for key in params.values()])

        while time() - self.last_request > self.delay:
            await asyncio.sleep(time() - self.last_request)

        self.last_request = time() + self.timeout
        async with self.session.get(self.API_SERVER + address, params=params, allow_redirects=False) as response:
            logger.debug(f'Successfully got {address} with {params} by request')
            self.last_request = time()
            return loads(await response.text())

    async def download(self, address: str, save_path: Path, params: dict | None = None) -> None:
        assert isinstance(self.session, ClientSession)
        assert isinstance(address, str)
        assert isinstance(save_path, Path)
        assert isinstance(params, dict) or params is None

        if params is None: params = dict()
        while (time() - self.last_request) < self.delay:
            await asyncio.sleep(time() - self.last_request)

        self.last_request: float = time() + self.timeout
        if address.startswith('https'):
            url: str = address
        else:
            url: str = self.API_SERVER + address
        try:
            async with self.session.get(url, allow_redirects=False, params=params) as response:
                logger.debug(f'Successfully downloaded {address} with {params}')
                self.last_request: float = time()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with save_path.open('wb+') as f: 
                    f.write(await response.read())
        except aiohttp.client_exceptions.ClientOSError:
            await asyncio.sleep(1)
            return await self.download(
                address=address,
                save_path=save_path,
                params=params
            )

    async def recheckAllVoices(
            self,
            bar: type[Bar] | None = None,
            bar_arguments: dict | None = None
        ) -> None:
        logger.info(f'Launching Downloader.recheckAllVoices(bar={bar})')
        assert bar is None or issubclass(bar, Bar)
        assert bar_arguments is None or isinstance(bar, dict)
        if bar_arguments is None: bar_arguments = dict()

        await self.updateInfo()
        await self.updateBasicServant()
        for i in range(1, self.basic_servant.collectionNoMax):
            voices = await ServantVoices.load(i)
            voices.buildVoiceLinesDict(fill_all_ascensions=False)
            await voices.updateVoices(bar=bar(**bar_arguments) if bar is not None else None)
        except:
            logger.warning(
                "Exception during Downloader.recheckAllVoices(). "
                f"Updated {i}/{self.basic_servant.collectionNoMax} servants"
            )
            raise

    def destroy(self) -> None:
        assert isinstance(self.session, ClientSession)
        asyncio.run(self.session.close())
        self.session = None


class VoiceLine:
    __slots__ = (
        '__dict__', 'dictionary'
    )
    dictionary: dict[str, Any]

    def __init__(self, values: dict[str, Any]) -> None:
        assert isinstance(values, dict)
        assert 'svt_id' in values, \
            "Servant id (svt_id) should be added in dictionary passed to VoiceLine"
        assert 'name' in values
        assert 'overwriteName' in values
        assert 'svtVoiceType' in values
        assert isinstance(values['name'], str)
        assert isinstance(values['overwriteName'], str)
        assert isinstance(values['svtVoiceType'], str)

        self.dictionary = values
        self.dictionary['name'] = self.dictionary['name']\
            .replace('{', '')\
            .replace('}', '')\
            .replace(':', ' -')\
            .replace('?', ' -')\
            .strip()
        self.dictionary['overwriteName'] = self.dictionary['overwriteName']\
            .replace('{', '')\
            .replace('}', '')\
            .replace(':', ' -')\
            .replace('?', ' -')\
            .strip()
        for i in ('name', 'overwriteName'):
            if '\r' in self.dictionary[i]:
                self.dictionary[i] = self.dictionary[i][:self.dictionary[i].find('\r')]

        """
        In events types, like
        "Land of Shadows Battleground Blitz - A Cat, a Bunny, and a Holy Grail War 6"
        we need to cut long strings. So, we finding dash and removing everything after it
        including dash
        Also, some events starts with "Revival -", so we must detect this
        """
        bracket_index = self.dictionary['overwriteName'].find('(')
        if bracket_index != -1:
            self.dictionary['overwriteName'] = self.dictionary['overwriteName'][:bracket_index]
        self.dictionary['overwriteName'] = '/'.join(
            i.strip() for i in self.dictionary['overwriteName'].split(' - ')
        )

    def __repr__(self) -> str:
        return f"<VL {self.name} for {self.ascension}>"

    @property
    def servant_id(self) -> int:
        return self.dictionary['svt_id']

    @cached_property
    def ascension(self) -> Ascension:
        id: str = self.dictionary['id'][0]
        return list(Ascension)[int(id[:id.find('_')])]

    @cached_property
    def type(self) -> VoiceLineCategory:
        return VoiceLineCategory.fromString(self.dictionary['svtVoiceType'])

    @cached_property
    def name(self) -> str:
        return self.dictionary['name']

    @cached_property
    def overwriteName(self) -> str:
        return self.dictionary['overwriteName']

    @cached_property
    def anyName(self) -> str:
        return self.overwriteName if self.overwriteName else self.name

    @cached_property
    def path_folder(self) -> Path:
        return Downloader.SERVANTS_FOLDER / str(self.servant_id) / Downloader.VOICES_FOLDER_NAME / \
            self.ascension.name / self.type.name / self.anyName

    @cached_property
    def filename(self) -> str:
        return f"{self.name if self.name else 'file'}.mp3"

    @cached_property
    def path(self) -> Path:
        return self.path_folder / self.filename

    @property
    def loaded(self) -> bool:
        return self.path.exists()

    def voiceLinesURL(self) -> Generator[str, None, None]:
        yield from self.dictionary['audioAssets']

    def concat_mp3(
        self,
        source_paths: list[Path]
    ) -> None:
        assert len({str(i.parent) for i in source_paths}) == 1  # Same parent folder
        assert not self.loaded
        assert source_paths

        filenames = [i.name for i in source_paths]

        command = (
            f"{Downloader.FFMPEG_PATH} -i \"concat:"
            f"{'|'.join(filenames)}"
            '" -c copy '
            f'"{self.filename}"'
        )
        ret = subprocess.call(
            args=command,
            cwd=self.path_folder,
            timeout=2,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if ret != 0: raise FFMPEGException("ffmpeg returned non-zero code")

        for source in source_paths:
            source.unlink()

    @staticmethod
    def leftovers_delete(paths: list[Path]) -> None:
        logger.warning('Exception during VoiceLine file download')
        if len(paths) == 0:
            return
        for path in paths:
            path.unlink(missing_ok=True)
        directory = paths[0]
        while True:
            directory = directory.parent
            if len(os.listdir(directory)) == 0:
                directory.rmdir()
                logger.info(f"Unlinked folder {directory.name} because it's empty")
            else:
                break
        logger.info('Unlinked temporary files successfully')

    async def download(self) -> None:
        downloader = Downloader()
        paths: list[Path] = []
        atexit.register(self.leftovers_delete, paths)
        for index, voice_url in enumerate(self.voiceLinesURL()):
            self.path_folder.mkdir(parents=True, exist_ok=True)
            paths.append(self.path)
            paths[-1] = paths[-1].with_stem(f"{paths[-1].stem}_{index}")
            if paths[-1].exists():
                getLogger('AA_voices_downloader').warning(
                    "Found out leftovers before audio concatenation: "
                    f'"{paths[-1].absolute()}". '
                    'This can be caused by program exit with error, or if file created by other process'
                )
            await downloader.download(voice_url, paths[-1])
        self.concat_mp3(paths)
        atexit.unregister(self.leftovers_delete)

    async def touch(self) -> None:
        assert self.loaded
        os.utime(self.path, (self.path.lstat().st_birthtime, time()))


class BasicServant:
    __slots__ = ('__dict__', 'values')

    def __init__(self, values: list) -> None:
        self.values: dict[int, dict] = {i['collectionNo']: i for i in values}

    def __getitem__(self, item: int) -> dict:
        assert isinstance(item, int)
        return self.values.__getitem__(item)

    @cached_property
    def collectionNoMax(self) -> int:
        return max(self.values.keys())

ANNOTATION_VOICE_CATEGORY = dict[VoiceLineCategory, dict[str, list[VoiceLine]]]
ANNOTATION_VOICES = dict[Ascension, ANNOTATION_VOICE_CATEGORY] | None
class ServantVoices:
    __slots__ = (
        'id', 'voice_lines', 'amount',
    )

    def __init__(self, id: int):
        self.id: int = id
        self.voice_lines: ANNOTATION_VOICES = None

    def __repr__(self) -> str:
        return f"<Svt {self.id}" + (' with voice lines' if self.voice_lines else '') + '>'

    @property
    def path(self) -> Path:
        return Downloader.SERVANTS_FOLDER / str(self.id)

    @property
    def path_json(self) -> Path:
        return self.path / 'info.json'

    @property
    def path_voices(self) -> Path:
        return self.path / 'voices'

    @classmethod
    async def get_json(cls, id: int) -> dict:
        json = await Downloader().request(
            address=f'/nice/NA/servant/{id}',
            params={'lore': 'true'}
        )
        assert isinstance(json, dict)
        return json


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
            break

        return ServantVoices(id=id)

    def buildVoiceLinesDict(self, fill_all_ascensions: bool = False) -> None:
        if not self.path_json.exists():
            raise RuntimeError("Load servants via ServantVoices.load() classmethod")
        data: dict = loads(self.path_json.read_text(encoding='utf-8'))
        output: ANNOTATION_VOICES = dict()
        self.amount = 0
        name_counters: dict[VoiceLineCategory, int] = dict()
        for voices in data['profile']['voices']:
            type = VoiceLineCategory.fromString(voices['type'])
            ascension = list(Ascension)[voices['voicePrefix']]
            if ascension not in output:
                output[ascension] = dict()
            if type not in output[ascension]:
                output[ascension][type] = dict()
            for line in voices['voiceLines']:
                if 'name' not in line:
                    continue
                name = line['name']
                if '{0}' in line['overwriteName']:
                    if name not in name_counters:
                        name_counters[name] = 0
                    else:
                        name_counters[name] += 1
                    line['overwriteName'] = line['overwriteName']\
                        .replace('{0}', str(name_counters[name]))
                if name not in output[ascension][type]:
                    output[ascension][type][name] = []
                line['svt_id'] = self.id
                output[ascension][type][name].append(VoiceLine(line))
                self.amount += 1

        if fill_all_ascensions:
            ascensionAdd = data['ascensionAdd']['voicePrefix']
            for ascension_str, target_ascension in ascensionAdd['ascension'].items():
                ascension_str: Ascension = list(Ascension)[int(ascension_str)]
                target_ascension: Ascension = list(Ascension)[target_ascension]
                assert target_ascension in output
                output[ascension_str] = output[target_ascension]

        self.voice_lines = output

    def loadedVoices(self) -> Generator[VoiceLine, None, None]:
        assert isinstance(self.voice_lines, dict)
        for ascension_values in self.voice_lines.values():
            for category_values in ascension_values.values():
                for type_values in category_values.values():
                    for voice_line in type_values:
                        if voice_line.loaded:
                            yield voice_line

    async def updateVoices(self, bar: Bar | None = None, message_size: int = 40) -> None:
        downloader = Downloader()
        if self.voice_lines is None:
            raise NoVoiceLines("Trying to updateVoices before buildVoiceLinesDict() called")
        if not hasattr(downloader, 'timestamps'):
            await downloader.updateInfo()
        if not self.path_json.exists():
            logger.info(f"S{self.id}: JSON doesn't exist, but must exist")
        elif downloader.timestamps['NA'] > self.path.lstat().st_mtime:
            logger.info(f'S{self.id}: folder modified before NA patch')
            current_json = loads(self.path_json.read_text(encoding='utf-8'))
            new_json = await self.get_json(self.id)
            if current_json != new_json:
                logger.info(f'S{self.id}: New JSON different from old')
                self.path_voices.unlink(missing_ok=True)
            else:
                os.utime(self.path, (self.path.lstat().st_birthtime, time()))
        logger.info(f'Started updating {self.id} voices')
        if bar is not None:
            voice_lines_amount = self.amount
            bar.max = voice_lines_amount
            bar.suffix = '%(index)d/%(max)d %(eta)ds'
            bar.message = 'Loading Servant info'
            bar.update()
        for ascension, ascension_values in self.voice_lines.items():
            for category_values in ascension_values.values():
                for type_values in category_values.values():
                    for voice_line in type_values:
                        if bar is not None:
                            bar.next()
                            bar.message = f"{ascension.name}: {voice_line.name: <30}"[:36]
                        if voice_line.loaded:
                            await voice_line.touch()
                            continue
                        await voice_line.download()
        if bar is not None:
            bar.message = f'Servant {self.id: >3} up-to-date'
            bar.suffix = '%(index)d/%(max)d'
            bar.update()
            bar.finish()
        logger.info(f'Finished updating {self.id} voices')
