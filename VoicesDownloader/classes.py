from io import StringIO
from typing import Any, Generator, Optional
import asyncio
import threading
import atexit
import os
import json
from logging import getLogger
import subprocess
from enum import IntEnum
from time import time
from time import sleep
from pathlib import Path
import shutil
from functools import cached_property

import aiohttp
from aiohttp import ClientSession
import aiohttp.client_exceptions
from progress.bar import Bar
from progress.spinner import Spinner

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


async def update_modified_date(path: Path):
    assert path.exists()
    os.utime(path, (path.lstat().st_birthtime, time()))


class NoVoiceLines(Exception): pass
class FFMPEGException(Exception): pass
class DownloadException(Exception): pass

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
    EventTowerReward = 11

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
            case 'eventTowerReward':
                return cls.EventTowerReward
            case _:
                raise Exception(f"There's no such category: \"{value}\"")


class ExceptionType(IntEnum):
    NP_IN_BATTLE_SECTION = 0
    SKIP_ON_DOWNLOAD_EXCEPTION = 1


MAINDIR = Path() / 'VoicesDownloader' / 'downloads'
SERVANT_EXCEPTIONS: dict[int, set[ExceptionType]] = {
    66: {ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION, },
    153: {ExceptionType.NP_IN_BATTLE_SECTION, },
    175: {ExceptionType.NP_IN_BATTLE_SECTION, },
    178: {ExceptionType.NP_IN_BATTLE_SECTION, },
    179: {ExceptionType.NP_IN_BATTLE_SECTION, },
    182: {ExceptionType.NP_IN_BATTLE_SECTION, },
    188: {ExceptionType.NP_IN_BATTLE_SECTION, },
    189: {ExceptionType.NP_IN_BATTLE_SECTION, }
}

class Downloader:
    __slots__ = (
        'delay', 'maximum_retries', 'timestamps', 'last_request',
        'session', 'basic_servant', 'animation_bool', 'animation_speed'
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

    def __init__(self, delay: float = 1, maximum_retries: int = 3) -> None:
        assert isinstance(delay, (int, float))
        assert isinstance(maximum_retries, int)
        if hasattr(self, 'delay'): return
        logger.info(f"Initialized Downloader with params {delay=}, {maximum_retries=}")
        self.delay = delay
        self.maximum_retries = maximum_retries
        self.last_request = time()
        self.session = ClientSession()
        self.animation_bool = False
        self.animation_speed = 0.5
        atexit.register(self.destroy)

    def _spinner_thread(self, spinner: Spinner):
        logger.debug("Spinner thread started")
        self.animation_bool = True
        while self.animation_bool:
            spinner.next()
            sleep(self.animation_speed)
        logger.debug("Spinner thread ended")

    async def updateInfo(self) -> None:
        info = await self.request_json('/info')
        assert isinstance(info, dict)
        self.timestamps = {region: data['timestamp'] for region, data in info.items() if region in {'NA', 'JP'}}
        assert len(self.timestamps) > 0

    async def updateBasicServant(self) -> None:
        json = await self.request_json('/export/NA/basic_servant.json')
        assert isinstance(json, list)
        self.basic_servant: BasicServant = BasicServant(json)

    async def request(self, address: str, params: dict | None = dict()) -> bytes:
        assert isinstance(self.session, ClientSession)
        assert isinstance(address, str)
        if params is not None:
            assert isinstance(params, dict)
            assert all([isinstance(key, (str, int, float)) for key in params.values()])
            assert all([isinstance(key, str) for key in params.keys()])

        if address.startswith('https'):
            url: str = address
        else:
            url: str = self.API_SERVER + address

        while (time() - self.last_request) < self.delay:
            await asyncio.sleep(time() - self.last_request)

        self.last_request = time() + self.delay
        calls_amount = 1
        while True:
            logger.debug(f'Request to {url} with {params}')
            try:
                async with self.session.get(url, params=params, allow_redirects=False) as response:
                    if calls_amount != 1:
                        logger.info(f"Successfully got {url} with {params} after {calls_amount-1} retries")
                    else:
                        logger.debug(f'Successfully got {url} with {params}')
                    self.last_request = time()
                    return await response.read()
            except aiohttp.client_exceptions.ClientError as e:
                logger.exception(
                    f'Caught aiohttp exception during request ({calls_amount}) '
                    f'to \"{url}\" with params={params},'
                    f'\tException {type(e).__name__}({e.args})',
                    exc_info=False
                )
                if calls_amount > self.maximum_retries:
                    logger.info(f'Reached maximum amount of retries ({calls_amount} > {self.maximum_retries})')
                    raise
                if calls_amount > 7:
                    await asyncio.sleep(7**3)
                else:
                    # 0 1 8 27 64 125 216 343 343 ...
                    await asyncio.sleep(calls_amount**3)
            calls_amount += 1

    async def request_json(self, address: str, params: dict = dict()) -> dict | list:
        return json.loads(await self.request(
            address=address,
            params=params
        ))

    async def download(self, address: str, save_path: Path, params: dict | None = None) -> None:
        assert isinstance(self.session, ClientSession)
        assert isinstance(address, str)
        assert isinstance(save_path, Path)
        assert isinstance(params, dict) or params is None

        save_path.parent.mkdir(parents=True, exist_ok=True)
        raw_json = await self.request(
            address=address,
            params=params
        )
        save_path.write_bytes(raw_json)

    async def recheckAllVoices(
            self,
            bar: type[Bar] | None = None,
            bar_arguments: dict | None = None,
            _spinner: type[Spinner] | None= None
        ) -> None:
        logger.info(f'Launching Downloader.recheckAllVoices(bar={bar})')
        assert bar is None or issubclass(bar, Bar)
        assert bar_arguments is None or isinstance(bar, dict)
        if bar_arguments is None: bar_arguments = dict()

        if _spinner is not None:
            spinner = _spinner(message='Updating info ')
            thread = threading.Thread(
                name='Spinner',
                daemon=True,
                target=self._spinner_thread,
                args=(spinner,)
            )
            thread.start()

        await self.updateInfo()
        if _spinner is not None: spinner.message = 'Updating Basic Servant '
        await self.updateBasicServant()
        if _spinner is not None:
            spinner.message = 'Finished '
            self.animation_bool = False
            thread.join()
            spinner.finish()
        try:
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
        for i in ('name', 'overwriteName'):
            self.dictionary[i] = self.dictionary[i]\
                .replace('{', '')\
                .replace('}', '')\
                .replace(':', ' -')\
                .replace('?', ' -')\
                .replace(' ☆', ' ')\
                .replace('☆ ', ' ')\
                .replace('☆', '')\
                .replace('\n', ' ')\
                .strip()
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
        splits = [i.strip() for i in self.dictionary['overwriteName'].split(' - ')]
        self.dictionary['path_add'] = '/'.join(splits[:-1])
        self.dictionary['overwriteName'] = splits[-1]
        self.dictionary['overwriteName'] = self.dictionary['overwriteName'].replace('/', ' ')

    def __repr__(self) -> str:
        return f"<VL {self.name} for {self.ascension}>"

    @property
    def servant_id(self) -> int:
        assert 'svt_id' in self.dictionary
        return self.dictionary['svt_id']

    @cached_property
    def ascension(self) -> Ascension:
        assert 'id' in self.dictionary
        id: str = self.dictionary['id'][0]
        return list(Ascension)[int(id[:id.find('_')])]

    @cached_property
    def type(self) -> VoiceLineCategory:
        assert 'svtVoiceType' in self.dictionary
        return VoiceLineCategory.fromString(self.dictionary['svtVoiceType'])

    @cached_property
    def name(self) -> str:
        assert 'name' in self.dictionary
        return self.dictionary['name']

    @cached_property
    def overwriteName(self) -> str:
        assert 'overwriteName' in self.dictionary
        return self.dictionary['overwriteName']

    @cached_property
    def anyName(self) -> str:
        return self.overwriteName if self.overwriteName else self.name

    @cached_property
    def path_folder(self) -> Path:
        return Downloader.SERVANTS_FOLDER / str(self.servant_id) / Downloader.VOICES_FOLDER_NAME / \
            self.ascension.name / self.type.name / self.dictionary['path_add']

    @cached_property
    def filename(self) -> str:
        index = '' if self.index == -1 else f" {self.index+1}"
        return f"{self.anyName}{index}.mp3"

    @cached_property
    def path(self) -> Path:
        return self.path_folder / self.filename

    @cached_property
    def index(self) -> int:
        assert isinstance(self.dictionary['_index'], int)
        return self.dictionary['_index']

    @property
    def loaded(self) -> bool:
        return self.path.exists()

    def _voiceLinesURL(self) -> Generator[str, None, None]:
        assert 'audioAssets' in self.dictionary
        yield from self.dictionary['audioAssets']

    @property
    def subtitle(self) -> str:
        assert 'subtitle' in self.dictionary
        return self.dictionary['subtitle']

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
        p = subprocess.Popen(
            args=command,
            cwd=self.path_folder,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output, err = p.communicate(timeout=10)
        ret = p.returncode
        if ret != 0:
            # May be server returned in response error?
            for i in source_paths:
                try:
                    loaded: dict = json.loads(i.read_bytes())
                except (
                    json.decoder.JSONDecodeError,
                    UnicodeDecodeError
                ):
                    continue
                for source in source_paths:
                    source.unlink()
                raise DownloadException(
                    f"File {self.path} couldn't be downloaded due to error received from AA:\n"
                    + json.dumps(loaded, indent=4)
                )

            raise FFMPEGException(
                "FFMpeg returned non-zero code. Leftovers left untouched. Additional info:"
                "\n> Command: "  + command +
                "\n> CWD: " + str(self.path_folder) + 
                "\n> StdOut:" +
                output.decode().replace('\n', '\n\t')
                +
                '\n> StdErr:' +
                err.decode().replace('\n', '\n\t')
            )

        for source in source_paths:
            source.unlink()

    @staticmethod
    def _leftovers_delete(paths: list[Path]) -> None:
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
        assert not self.path.exists()
        downloader = Downloader()
        paths: list[Path] = []
        atexit.register(self._leftovers_delete, paths)
        for index, voice_url in enumerate(self._voiceLinesURL()):
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
        try:
            self.concat_mp3(paths)
        except FFMPEGException:
            atexit.unregister(self._leftovers_delete)
            raise
        else:
            atexit.unregister(self._leftovers_delete)


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
        'id', 'voice_lines', 'amount', 'skipped_amount'
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
    async def _get_json(cls, id: int) -> dict:
        json = await Downloader().request_json(
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

        sv = ServantVoices(id=id)
        asyncio.create_task(update_modified_date(sv.path_json))
        return sv

    def buildVoiceLinesDict(self, fill_all_ascensions: bool = False) -> None:
        if not self.path_json.exists():
            raise RuntimeError("Load servants via ServantVoices.load() classmethod")
        data: dict = json.loads(self.path_json.read_text(encoding='utf-8'))
        output: ANNOTATION_VOICES = dict()
        self.amount = 0
        self.skipped_amount = 0
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
                name = line['overwriteName'] if line['overwriteName'] else line['name']
                if '{0}' in line['overwriteName']:
                    if name not in name_counters:
                        name_counters[name] = 0
                    else:
                        name_counters[name] += 1
                    line['overwriteName'] = line['overwriteName']\
                        .replace('{0}', str(name_counters[name]))
                if name not in output[ascension][type]:
                    output[ascension][type][name] = []
                    line['_index'] = -1
                else:
                    output[ascension][type][name][-1]\
                        .dictionary['_index'] = len(output[ascension][type][name])-1
                    line['_index'] = len(output[ascension][type][name])
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

        exceptions = SERVANT_EXCEPTIONS.get(self.id, set())
        if exceptions:
            if ExceptionType.NP_IN_BATTLE_SECTION in exceptions:
                for ascension_values in self.voice_lines.values():
                    category = ascension_values[VoiceLineCategory.Battle]
                    to_pop = []
                    for name, voice_lines in category.items():
                        if all((
                            'Noble Phantasm' in name,
                            'Card' not in name
                        )):
                            to_pop.append(name)
                            self.amount -= 1
                            self.skipped_amount += 1
                            logger.warning(
                                f"S{self.id}: Skipped {self.path} because NP card broken"
                            )
                    for name in to_pop: category.pop(name)

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
            logger.exception(f"S{self.id}: JSON doesn't exist, but must exist")
        elif downloader.timestamps['NA'] > self.path.lstat().st_mtime:
            logger.info(f'S{self.id}: folder modified before NA patch')
            current_json = json.loads(self.path_json.read_text(encoding='utf-8'))
            new_json = await self._get_json(self.id)
            if current_json != new_json:
                logger.info(f'S{self.id}: New JSON different from old')
                shutil.rmtree(self.path_voices)
            else:
                asyncio.create_task(update_modified_date(self.path))
        logger.debug(f'Started updating {self.id} voices')
        self.path_voices.mkdir(parents=False, exist_ok=True)
        if bar is not None:
            voice_lines_amount = self.amount
            bar.max = voice_lines_amount
            bar.suffix = '%(index)d/%(max)d %(eta)ds'
            bar.message = 'Loading Servant info'
            bar.update()
        downloaded_counter = 0
        for ascension_values in self.voice_lines.values():
            for category_values in ascension_values.values():
                for type_values in category_values.values():
                    for voice_line in type_values:
                        if bar is not None:
                            bar.message = (f"{voice_line.ascension.name}: " +
                                voice_line.name.__format__(f" <{message_size-11}")
                            )[:message_size]
                            bar.next()
                        if voice_line.loaded:
                            continue
                        downloaded_counter += 1
                        try:
                            await voice_line.download()
                        except DownloadException:
                            if ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION\
                                not in\
                                SERVANT_EXCEPTIONS[self.id]:
                                raise
                            self.skipped_amount += 1
                            logger.warning(
                                f"S{self.id}: Skipping VoiceLine {voice_line.path} due to"
                                f" {ExceptionType.__name__}."
                                f"{ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION.name}"
                                " == true"
                            )
                            continue

                        asyncio.create_task(update_modified_date(self.path))
        if bar is not None:
            bar.message = f'Servant {self.id: >3} up-to-date'
            bar.suffix = '%(index)d/%(max)d'
            downloaded = f' (downloaded: {downloaded_counter})' if downloaded_counter else ''
            skipped = f' (skipped: {self.skipped_amount})' if self.skipped_amount else ''
            bar.suffix = bar.suffix.ljust(7) + downloaded + skipped
            bar.update()
            bar.finish()
        if downloaded_counter:
            logger.info(f"Downloaded {downloaded_counter}/{voice_lines_amount} "
                f"for servant {self.id} (skipped {self.skipped_amount})"
            )
        asyncio.create_task(update_modified_date(self.path_voices))
        logger.debug(f'Finished updating {self.id} voices')
