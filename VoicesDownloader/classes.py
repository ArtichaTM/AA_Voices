from typing import Any, AsyncGenerator, Callable, Generator
import asyncio
import threading
import atexit
import os
import json
import subprocess
from logging import getLogger
from enum import IntEnum
from time import time, sleep
from pathlib import Path
import shutil

import aiohttp
import aiohttp.client_exceptions
import eyed3.id3
from progress.bar import Bar
from progress.spinner import Spinner
import eyed3

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
    """ Set's path modify time to now
    :param path: Target path
    """
    assert path.exists()
    os.utime(path, (path.lstat().st_birthtime, time()))


class NoVoiceLines(Exception):
    """Raised when ServantVoices used without initiating via buildVoiceLinesDict()"""
class NoSuchCategory(Exception):
    """Raised when ServantVoices used without initiating via buildVoiceLinesDict()"""
class FFMPEGException(Exception):
    """Raised when FFMpeg launch returns non-zero code (when some fault occurred)"""
class DownloadException(Exception):
    """Raised when downloading file impossible or error occurred during download"""

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
    """Used to replace category's with numbers"""
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
    EventDailyPoint = 12
    TreasureBox = 13
    EventDigging = 14

    @classmethod
    def fromString(cls, value: str) -> 'VoiceLineCategory':
        """ Converts string to VoiceLineCategory
        :param value: String containing name of category
        :raises NoSuchCategory: Raised when no such category existing
        :return: Category as class instance
        """
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
            case 'eventDailyPoint':
                return cls.EventDailyPoint
            case 'treasureBox':
                return cls.TreasureBox
            case 'eventDigging':
                return cls.EventDigging
            case _:
                raise Exception(f"There's no such category: \"{value}\"")


class ExceptionType(IntEnum):
    NP_IN_BATTLE_SECTION = 0
    SKIP_ON_DOWNLOAD_EXCEPTION = 1


class PropertyOneCall[T]:
    def __init__(self, fget: Callable[[Any], T] | None =None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc
        self._name = ''

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, *_) -> T:
        if self.fget is None:
            raise AttributeError(
                f'property {self._name!r} of {type(obj).__name__!r} object has no getter'
             )
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError(
                f'property {self._name!r} of {type(obj).__name__!r} object has no setter'
             )
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError(
                f'property {self._name!r} of {type(obj).__name__!r} object has no deleter'
             )
        self.fdel(obj)

    def getter(self, fget):
        prop = type(self)(fget, self.fset, self.fdel, self.__doc__)
        prop._name = self._name
        return prop

    def setter(self, fset):
        prop = type(self)(self.fget, fset, self.fdel, self.__doc__)
        prop._name = self._name
        return prop

    def deleter(self, fdel):
        prop = type(self)(self.fget, self.fset, fdel, self.__doc__)
        prop._name = self._name
        return prop


class DeepComparer:
    __slots__ = ('left', 'right')

    class ComparerException(Exception):
        """
        Base exception for all DeepComparer exceptions
        Arguments: reaching_keys: list, *
        """
        def _path(self) -> str:
            assert isinstance(self.args[0], list)
            return f"values[{']['.join((str(i) for i in self.args[0]))}]"
        def prettify(self) -> str:
            raise NotImplementedError()

    class DifferentLength(ComparerException):
        """
        Collections in the same position have different size.
        """
        def prettify(self) -> str:
            assert isinstance(self.args[0], list)
            assert isinstance(self.args[1], int)
            assert isinstance(self.args[2], int)
            return f"In {self._path()} {self.args[1]} != {self.args[2]}"

    class DifferentTypes(ComparerException):
        """
        Objects in the same position have different types.
        """
        def prettify(self) -> str:
            assert isinstance(self.args[0], list)
            assert isinstance(self.args[1], type)
            assert isinstance(self.args[2], type)
            return f"In {self._path()} {self.args[1]} != {self.args[2]}"

    class DifferentDictKeys(ComparerException):
        """
        Objects in the same position have different keys.
        """
        def prettify(self) -> str:
            assert isinstance(self.args[0], list)
            return f"In {self._path()} {self.args[1]} != {self.args[2]}"

    class DifferentValues(ComparerException):
        """
        Objects in the same position have different values and they are not dict|list.
        """
        def prettify(self) -> str:
            assert isinstance(self.args[0], list)
            return f"In {self._path()} {self.args[1]} != {self.args[2]}"


    def __init__(self, left: dict | list, right: dict | list) -> None:
        assert isinstance(left, (dict, list)), 'Only list or dict supported'
        assert isinstance(right, (dict, list)), 'Only list or dict supported'
        assert isinstance(left, type(right)), 'Right and red must be same type'
        self.left = left
        self.right = right

    @classmethod
    def _deepCompareList(cls, left: list, right: list) -> None:
        if len(left) != len(right):
            raise cls.DifferentLength([], len(left), len(right))
        for index, (lv, rv) in enumerate(zip(left, right)):
            if isinstance(lv, list):
                if not isinstance(rv, list):
                    raise cls.DifferentTypes([index])
                method = cls._deepCompareList
            elif isinstance(lv, dict):
                if not isinstance(rv, dict):
                    raise cls.DifferentTypes([index])
                method = cls._deepCompareDict
            else:
                if lv != rv:
                    raise cls.DifferentValues([index])
            try:
                method(lv, rv)
            except cls.ComparerException as e:
                assert isinstance(e.args[0], list)
                e.args[0].append(index)
                raise

    @classmethod
    def _deepCompareDict[T](
        cls,
        left: dict[T, Any],
        right: dict[T, Any]
    ) -> None:
        if len(left) != len(right):
            raise cls.DifferentLength(len(left), len(right))
        for (lk, lv), (rk, rv) in zip(left.items(), right.items()):
            assert not isinstance(lk, (dict, list)), "How?"
            assert not isinstance(rk, (dict, list)), "How?"
            if lk != rk:
                raise cls.DifferentDictKeys([], lk, rk)
            if lv != rv:
                raise cls.DifferentValues([lk], lv, rv)

    def compare(self) -> None:
        if isinstance(self.left, dict):
            assert isinstance(self.right, dict)
            method = self._deepCompareDict
        else:
            assert isinstance(self.left, list)
            assert isinstance(self.right, list)
            method = self._deepCompareList
        try:
            method(self.left, self.right)
        except self.ComparerException as e:
            assert isinstance(e.args[0], list)
            e.args[0].reverse()

ALBUM_NAME = 'Fate: Grand Order Servants'
MAINDIR = Path() / 'dataset'
SERVANT_EXCEPTIONS: dict[int, set[ExceptionType]] = {
    66: {ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION, }
    , 153: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 175: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 177: {ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION, }
    , 178: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 179: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 182: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 188: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 189: {ExceptionType.NP_IN_BATTLE_SECTION, }
    , 205: {ExceptionType.NP_IN_BATTLE_SECTION, }
}

class Downloader:
    """ Class for downloading data from atlas academy """
    __slots__ = (
        'delay', 'maximum_retries', 'timestamps', 'last_request',
        'session', 'basic_servant', 'animation_bool', 'animation_speed'
    )
    _instance: 'Downloader | None' = None
    API_SERVER: str = r'https://api.atlasacademy.io'
    SERVANTS_FOLDER = MAINDIR / 'Servants'
    VOICES_FOLDER_NAME = 'voices'
    FFMPEG_PATH: str = 'ffmpeg'

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance: Downloader | None = super().__new__(cls)
        return cls._instance

    def __init__(self, delay: float = 1, maximum_retries: int = 3) -> None:
        assert isinstance(delay, (int, float))
        assert isinstance(maximum_retries, int)
        if hasattr(self, 'delay'): return
        logger.info(f"Initialized Downloader with params {delay=}, {maximum_retries=}")
        self.delay: float | int = delay
        self.maximum_retries: int = maximum_retries
        self.last_request: float = time()
        self.session: aiohttp.ClientSession | None = aiohttp.ClientSession()
        self.animation_bool: bool = False
        self.animation_speed: float | int = 0.5
        self.timestamps: dict[str, float] | None = None
        self.basic_servant: BasicServant | None = None
        atexit.register(self.destroy)

    def _spinner_thread(self, spinner: Spinner):
        """ Function for updating spinner in other thread
        Sets self.animation_bool to True automatically
        Relies on self.animation_speed
        :param spinner: Target spinner
        """
        assert self.session is not None, "Instance destroyed"
        logger.debug("Spinner thread started")
        self.animation_bool = True
        while self.animation_bool:
            spinner.next()
            sleep(self.animation_speed)
        logger.debug("Spinner thread ended")

    async def servants(
            self,
            buildVoiceLines: bool = True,
            updateVoices: bool = False
        ) -> AsyncGenerator['ServantVoices', None]:
        """ Prints in stdin all path conflicts for all servants """
        assert self.session is not None, "Instance destroyed"
        await self.updateBasicServant()
        assert isinstance(self.basic_servant, BasicServant)
        for i in range(1, self.basic_servant.collectionNoMax+1):
            voices = await ServantVoices.load(i)
            if buildVoiceLines:
                voices.buildVoiceLinesDict(fill_all_ascensions=False)
            if updateVoices:
                await voices.updateVoices()
            yield voices

    async def _print_all_conflicts(self) -> None:
        """ Prints in stdin all path conflicts for all servants """
        async for servant in self.servants():
            servant._print_conflicts()

    async def updateInfo(self) -> None:
        """
        Sets self.timestamps.
        Can be called only once (further calls makes no effect)
        """
        assert self.session is not None, "Instance destroyed"
        if self.timestamps is not None:
            return
        info = await self.request_json('/info')
        assert isinstance(info, dict)
        self.timestamps = {region: data['timestamp'] for region, data in info.items() if region in {'NA', 'JP'}}
        assert len(self.timestamps) > 0

    async def updateBasicServant(self) -> None:
        """
        Sets self.basic_servant
        Can be called only once (further calls makes no effect)
        """
        assert self.session is not None, "Instance destroyed"
        if self.basic_servant is not None:
            return
        json = await self.request_json('/export/NA/basic_servant.json')
        assert isinstance(json, list)
        self.basic_servant = BasicServant(json)

    async def request(self, address: str, params: dict | None = dict()) -> bytes:
        """ Requests data from atlas academy API
        :param address: Address without domain name like in rapidoc documentation
            example: "/basic/{region}/servant/{servant_id}"
        :param params: parameters passed to request.
            Always "lore=True" for servant json requests.
            Defaults to dict()
        :return: bytes received from API. No validation made
        """
        assert self.session is not None, "Instance destroyed"
        assert isinstance(self.session, aiohttp.ClientSession)
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
        """ Usual request(), but this returns json instead of bytes """
        assert self.session is not None, "Instance destroyed"
        return json.loads(await self.request(
            address=address,
            params=params
        ))

    async def download(self, address: str, save_path: Path, params: dict | None = None) -> None:
        """ Downloads any file from atlas academy api (site check included)
        :param address: Address passed to request()
        :param save_path: Target path where write bytes
        :param params: Parameters passed to request()
        """
        assert self.session is not None, "Instance destroyed"
        assert isinstance(self.session,aiohttp.ClientSession)
        assert isinstance(address, str)
        assert address.startswith('/') or address.startswith('https://static.atlasacademy.io'),\
            f"Address {address} request data from unknown site"
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
            spinner: type[Spinner] | None= None
        ) -> None:
        """ Starts job to check all missing voice lines and their download
        :param bar: Bar class to track current progress.
            If None progress won't be printed
        :param bar_arguments: Arguments to pass to Bar instance
        :param _spinner: _description_, defaults to None
        :raises DownloadException: _description_
        :raises FFMPEGException: _description_
        """
        assert self.session is not None, "Instance destroyed"
        assert bar is None or issubclass(bar, Bar)
        assert bar_arguments is None or isinstance(bar, dict)
        logger.info(f'Launching Downloader.recheckAllVoices(bar={bar})')
        if bar_arguments is None: bar_arguments = dict()

        if spinner is not None:
            spin = spinner(message='Updating info ')
            thread = threading.Thread(
                name='Spinner',
                daemon=True,
                target=self._spinner_thread,
                args=(spin,)
            )
            thread.start()

        await self.updateInfo()
        assert isinstance(self.timestamps, dict)
        if spinner is not None: spin.message = 'Updating Basic Servant '
        await self.updateBasicServant()
        assert isinstance(self.basic_servant, BasicServant)
        if spinner is not None:
            spin.message = 'Finished '
            self.animation_bool = False
            thread.join()
            spin.finish()
        try:
            async for servant in self.servants():
                await servant.updateVoices(bar=bar(**bar_arguments) if bar is not None else None)
        except:
            logger.warning(
                "Exception during Downloader.recheckAllVoices(). "
                f"Updated {servant.collectionNo}/{self.basic_servant.collectionNoMax} servants"
            )
            raise

    def destroy(self) -> None:
        """ Deletes current instance """
        assert isinstance(self.session, aiohttp.ClientSession)
        asyncio.run(self.session.close())
        self.session = None
        type(self)._instance = None


class VoiceLine:
    """
    Container class to hold information about voice line.
    On initializing edits some voice_line parameters.
    Dictionary values can be changed to alter voice_line properties returns.
        Example: change VoiceLine.dictionary['ascension'] to change ascension of voice line
    All properties lazy and cached to ensure non-repeating context
    """
    __slots__ = (
        '__dict__', 'dictionary'
    )

    def __init__(self, values: dict[str, Any]) -> None:
        """ Class for containing dictionary about voice line """
        assert isinstance(values, dict)
        assert 'svt_id' in values, \
            "Servant id (svt_id) should be added in dictionary passed to VoiceLine"
        assert 'name' in values
        assert 'overwriteName' in values
        assert 'svtVoiceType' in values
        assert 'svt_name' in values
        assert isinstance(values['name'], str)
        assert isinstance(values['overwriteName'], str)
        assert isinstance(values['svtVoiceType'], VoiceLineCategory)

        self.dictionary: dict[str, Any] = values
        for i in ('name', 'overwriteName'):
            self.dictionary[i] = self.dictionary[i]\
                .replace(' \r\n- ', ' - ')\
                .replace('\r\n', ' - ')\
                .replace('\r', ' ')\
                .replace('{', '')\
                .replace('}', '')\
                .replace('"', '_')\
                .replace("'", '_')\
                .replace(':', ' -')\
                .replace('?', ' -')\
                .replace('☆', '')\
                .strip()

        """
        In events types, like
        "Land of Shadows Battleground Blitz - A Cat, a Bunny, and a Holy Grail War 6"
        we need to cut long strings. So, we finding dash and removing everything after it
        including dash
        Also, some events starts with "Revival -", so we must detect this
        """
        splits = [
            i.strip() for i in self.dictionary['overwriteName']
                .replace('\n', ' - ')
                .split(' - ')
        ]
        self.dictionary['path_add'] = '/'.join(splits[:-1])
        self.dictionary['overwriteName'] = splits[-1]
        self.dictionary['overwriteName'] = self.dictionary['overwriteName'].replace('/', ' ')

    def __repr__(self) -> str:
        return f"<VL {self.name} for {self.ascension}>"

    @PropertyOneCall
    def servant_id(self) -> int:
        assert 'svt_id' in self.dictionary
        return self.dictionary['svt_id']

    @PropertyOneCall
    def servant_name(self) -> int:
        assert 'svt_name' in self.dictionary
        return self.dictionary['svt_name']

    @PropertyOneCall
    def ascension(self) -> Ascension:
        assert 'ascension' in self.dictionary
        return self.dictionary['ascension']

    @PropertyOneCall
    def type(self) -> VoiceLineCategory:
        assert 'svtVoiceType' in self.dictionary
        assert isinstance(self.dictionary['svtVoiceType'], VoiceLineCategory)
        return self.dictionary['svtVoiceType']

    @PropertyOneCall
    def name(self) -> str:
        assert 'name' in self.dictionary
        return self.dictionary['name']

    @PropertyOneCall
    def overwriteName(self) -> str:
        assert 'overwriteName' in self.dictionary
        return self.dictionary['overwriteName']

    @PropertyOneCall
    def anyName(self) -> str:
        """ Guarantee to return any name string """
        return self.overwriteName if self.overwriteName else self.name

    @PropertyOneCall
    def path_folder(self) -> Path:
        """ Path to folder containing voice line """
        return Downloader.SERVANTS_FOLDER / str(self.servant_id) / Downloader.VOICES_FOLDER_NAME / \
            self.ascension.name / self.type.name / self.dictionary['path_add']

    @PropertyOneCall
    def filename(self) -> str:
        """ Full file name in the destination folder. Example: "Skill 1.mp4" """
        index = '' if self.index == -1 else f" {self.index+1}"
        return f"{self.anyName}{index}.mp3"

    @PropertyOneCall
    def path(self) -> Path:
        """ Full path to voice line"""
        return self.path_folder / self.filename

    @PropertyOneCall
    def index(self) -> int:
        """ Index of the voice line. index != -1 only if "{0}" in overwriteName """
        assert isinstance(self.dictionary['_index'], int)
        return self.dictionary['_index']

    @property
    def loaded(self) -> bool:
        """ Returns True if voice line downloaded """
        return self.path.exists()

    @PropertyOneCall
    def subtitle(self) -> str:
        """ Text of the voice line [ENG]"""
        assert 'subtitle' in self.dictionary
        self.dictionary['subtitle'] = self.dictionary['subtitle'].split(']')[-1]
        return self.dictionary['subtitle']

    def _voiceLinesURL(self) -> Generator[str, None, None]:
        """ Iterator over URLs to voice line parts"""
        assert 'audioAssets' in self.dictionary
        yield from self.dictionary['audioAssets']

    async def metadata_update(self) -> None:
        downloader = Downloader()
        _d3: eyed3.AudioFile | None = eyed3.load(str(self.path))

        if downloader.basic_servant is None:
            await downloader.updateBasicServant()
        assert isinstance(downloader.basic_servant, BasicServant)
        assert _d3 is not None
        assert isinstance(_d3.tag, eyed3.id3.tag.Tag)
        assert _d3.tag.comments is not None

        tag = _d3.tag
        tag.artist = self.servant_name
        tag.album = ALBUM_NAME
        tag.title = self.name
        assert tag.comments is not None
        tag.comments.set(self.subtitle)
        tag.save()


    def concat_mp3(
        self,
        source_paths: list[Path]
    ) -> None:
        """ Generates current voice_line from sources
        When FFMpeg fails, method checks source files for JSON readability.
        If method succeeds in reading, DownloadException raised
        :param source_paths: List of paths to concat. MUST be in the same folder
        :raises DownloadException: At least one of the sources failed to download
        :raises FFMPEGException: Raised when all files OK but FFMpeg failed
        """
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
        """ Removes any leftovers. First of all, all paths unlinked.
        Then, if parent folder empty, unlink it as well
        """
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
        """ Downloads current voice line.
            Uses Downloader, self.path/_leftovers_delete/_voiceLinesURL/concat_mp3
        """
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
        asyncio.create_task(self.metadata_update())


class BasicServant:
    """ Contains basic information about all servants """
    __slots__ = ('__dict__', 'values')

    def __init__(self, values: list) -> None:
        """
        :param values: Parsed JSON from data
        """
        self.values: dict[int, dict] = {i['collectionNo']: i for i in values}

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, item: int) -> dict:
        assert isinstance(item, int)
        return self.values.__getitem__(item)

    @property
    def collectionNoMax(self) -> int:
        """ Latest servant index of all servants amount """
        return max(self.values.keys())


ANNOTATION_VOICE_CATEGORY = dict[VoiceLineCategory, dict[str, list[VoiceLine]]]
ANNOTATION_VOICES = dict[Ascension, ANNOTATION_VOICE_CATEGORY] | None
class ServantVoices:
    """ Container of VoiceLine-s"""
    __slots__ = (
        'collectionNo', 'voice_lines', 'amount',
        'skipped_amount', '_dictionary', '_name_overwrites'
    )

    def __init__(self, collectionNo: int):
        self.collectionNo: int = collectionNo
        self.voice_lines: ANNOTATION_VOICES = None
        self._name_overwrites: dict[Ascension, str] | None = None
        self._dictionary: dict | None = None

    def __repr__(self) -> str:
        return f"<Svt {self.collectionNo}" + (' with voice lines' if self.voice_lines else '') + '>'

    @PropertyOneCall
    def path(self) -> Path:
        """ Path to servant folder, containing JSON and voices folder"""
        return Downloader.SERVANTS_FOLDER / str(self.collectionNo)

    @PropertyOneCall
    def path_json(self) -> Path:
        """ Path to servant JSON """
        return self.path / 'info.json'

    @PropertyOneCall
    def path_voices(self) -> Path:
        """ Path to servant voices folder (contains voice lines)"""
        return self.path / 'voices'

    def name(self, ascension: Ascension) -> str:
        assert isinstance(self._dictionary, dict)
        assert isinstance(self._name_overwrites, dict)
        return self._name_overwrites.get(ascension, self._dictionary['name'])

    @classmethod
    async def _get_json(cls, collectionNo: int) -> dict:
        """ Download and parse current servant JSON info
        :param id: collectionNo of servant
        :type id: int
        :return: parsed JSON
        """
        json = await Downloader().request_json(
            address=f'/nice/NA/servant/{collectionNo}',
            params={'lore': 'true'}
        )
        assert isinstance(json, dict)
        return json

    @classmethod
    async def _updateJSON(cls, collectionNo: int) -> None:
        """ Updates (replaces) current servant JSON """
        servant_folder = Downloader.SERVANTS_FOLDER / f"{collectionNo}"
        servant_json_path = servant_folder / 'info.json'
        servant_json_path.parent.mkdir(parents=True, exist_ok=True)

        await Downloader().download(
            address=f'/nice/NA/servant/{collectionNo}',
            save_path=servant_json_path,
            params={'lore': 'true'}
        )

    @classmethod
    async def load(cls, collectionNo: int) -> 'ServantVoices':
        """ Loads servant with specific collectionNo
        :param id: _description_
        :type id: int
        :return: _description_
        """
        servant_folder = Downloader.SERVANTS_FOLDER / f"{collectionNo}"
        servant_json_path = servant_folder / 'info.json'

        while True:
            # JSON doesn't exist
            if not servant_json_path.exists():
                await cls._updateJSON(collectionNo=collectionNo)
            break

        sv = ServantVoices(collectionNo=collectionNo)
        asyncio.create_task(update_modified_date(sv.path_json))
        return sv

    def buildVoiceLinesDict(self, fill_all_ascensions: bool = False) -> None:
        """ This method sole target: make self.voice_lines. But it's a complex task
        :param fill_all_ascensions: If True, all self.voice_lines will be filled with ascensions 0-4
        :raises RuntimeError: When .buildVoiceLinesDict() used without loading Servant via .load()
        """
        if not self.path_json.exists():
            raise RuntimeError("Load servants via ServantVoices.load() classmethod")
        self._dictionary: dict | None = json.loads(self.path_json.read_text(encoding='utf-8'))
        assert isinstance(self._dictionary, dict)

        assert 'ascensionAdd' in self._dictionary
        assert isinstance(self._dictionary['ascensionAdd'], dict)
        assert 'overWriteServantName' in self._dictionary['ascensionAdd']
        assert isinstance(self._dictionary['ascensionAdd']['overWriteServantName'], dict)
        assert self._name_overwrites is None
        self._name_overwrites = dict()
        for type in ('ascension', 'costume'):
            for number, overwrite_name in self._dictionary['ascensionAdd']['overWriteServantName'][type].items():
                assert isinstance(number, str)
                assert isinstance(overwrite_name, str)
                self._name_overwrites[Ascension(int(number))] = overwrite_name

        output: ANNOTATION_VOICES = dict()
        self.amount = 0
        self.skipped_amount = 0
        name_counters: dict[VoiceLineCategory, int] = dict()
        for voices in self._dictionary['profile']['voices']:
            type = VoiceLineCategory.fromString(voices['type'])
            ascension = list(Ascension)[voices['voicePrefix']]
            if ascension not in output:
                output[ascension] = dict()
            if type not in output[ascension]:
                output[ascension][type] = dict()
            for line in voices['voiceLines']:
                line['ascension'] = ascension
                line['svtVoiceType'] = type
                line['svt_name'] = self.name(ascension)
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
                line['svt_id'] = self.collectionNo
                output[ascension][type][name].append(VoiceLine(line))
                self.amount += 1

        if fill_all_ascensions:
            ascensionAdd = self._dictionary['ascensionAdd']['voicePrefix']
            for ascension_str, target_ascension in ascensionAdd['ascension'].items():
                ascension_str: Ascension = list(Ascension)[int(ascension_str)]
                target_ascension: Ascension = list(Ascension)[target_ascension]
                assert target_ascension in output
                output[ascension_str] = output[target_ascension]

        self.voice_lines = output

        exceptions = SERVANT_EXCEPTIONS.get(self.collectionNo, set())
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
                                f"S{self.collectionNo}: Skipped {self.path} because NP card broken"
                            )
                    for name in to_pop: category.pop(name)

    def allVoices(self) -> Generator[VoiceLine, None, None]:
        """ Iterates over all possible, correct voice lines """
        assert isinstance(self.voice_lines, dict)
        for ascension_values in self.voice_lines.values():
            for category_values in ascension_values.values():
                for type_values in category_values.values():
                    for voice_line in type_values:
                        yield voice_line

    def loadedVoices(self) -> Generator[VoiceLine, None, None]:
        """ Iterates over all possible, correct and downloaded voice lines"""
        for voice_line in self.allVoices():
            if voice_line.loaded:
                yield voice_line

    def _iterate_over_conflicts(self) -> Generator[list[VoiceLine], None, None]:
        """ Yields list of voice lines that points to the same path """
        if self.voice_lines is None:
            raise NoVoiceLines("Trying to updateVoices before buildVoiceLinesDict() called")
        paths: dict[str, list[VoiceLine]] = dict()
        duplicates: set[str] = set()
        for voice_line in self.allVoices():
            path = str(voice_line.path)
            if path not in paths:
                paths[path] = [voice_line]
            else:
                duplicates.add(path)
                paths[path].append(voice_line)
        for duplicate in duplicates:
            yield paths[duplicate]


    def _print_conflicts(self) -> None:
        """ Using _iterate_over_conflicts() to print conflicts """
        for duplicates in self._iterate_over_conflicts():
            print(
                f"{self.collectionNo: >3}: "
                f"{', '.join((f"{i.name} ({i.type})".ljust(50) for i in duplicates))}"
                f" points to one path {duplicates[0].path}"
            )


    async def updateVoices(self, bar: Bar | None = None, message_size: int = 40) -> None:
        """ Main possible function for VoiceLine. Downloading current voice line
            with tracking progress with created Bar.
        :param bar: Created bar to track progress. None, if no progress track needed
        :param message_size: Bar.message size. No effect if bar=None
        :raises NoVoiceLines: When buildVoiceLinesDict() did not call before updateVoices()
        """
        downloader = Downloader()
        if self.voice_lines is None:
            raise NoVoiceLines("Trying to updateVoices before buildVoiceLinesDict() called")
        if downloader.timestamps is None:
            await downloader.updateInfo()
        assert isinstance(downloader.timestamps, dict)
        if not self.path_json.exists():
            logger.exception(f"S{self.collectionNo}: JSON doesn't exist, but must exist")
        elif downloader.timestamps['NA'] > self.path.lstat().st_mtime:
            logger.info(f'S{self.collectionNo}: folder modified before NA patch')
            current_json = json.loads(self.path_json.read_text(encoding='utf-8'))
            new_json = await self._get_json(self.collectionNo)
            comparer = DeepComparer(current_json, new_json)
            try:
                comparer.compare()
            except comparer.ComparerException as e:
                logger.info(f'S{self.collectionNo}: New JSON different from old. Where: {e.prettify()}')
                shutil.rmtree(self.path_voices)
            else:
                logger.info(f'S{self.collectionNo}: New JSON same as old')
                asyncio.create_task(update_modified_date(self.path))
        logger.debug(f'Started updating {self.collectionNo} voices')
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
                            bar.suffix = '%(index)d/%(max)d'
                            downloaded = f' (downloaded: {downloaded_counter})' if downloaded_counter else ''
                            skipped = f' (skipped: {self.skipped_amount})' if self.skipped_amount else ''
                            bar.suffix = bar.suffix.ljust(7) + downloaded + skipped
                            bar.message = (f"{voice_line.ascension.name}: " +
                                voice_line.name.__format__(f" <{message_size-11}")
                            )[:message_size]
                            bar.index += 1
                        if voice_line.loaded:
                            continue
                        downloaded_counter += 1
                        try:
                            await voice_line.download()
                        except DownloadException:
                            if ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION\
                                not in\
                                SERVANT_EXCEPTIONS.get(self.collectionNo, set()):
                                raise
                            self.skipped_amount += 1
                            downloaded_counter  -= 1
                            logger.warning(
                                f"S{self.collectionNo}: Skipping VoiceLine {voice_line.path} due to"
                                f" {ExceptionType.__name__}."
                                f"{ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION.name}"
                                " == true"
                            )
                        if bar is not None:
                            bar.update()

                        asyncio.create_task(update_modified_date(self.path))
        if bar is not None:
            bar.message = f'Servant {self.collectionNo: >3} up-to-date'
            bar.suffix = '%(index)d/%(max)d'
            downloaded = f' (downloaded: {downloaded_counter})' if downloaded_counter else ''
            skipped = f' (skipped: {self.skipped_amount})' if self.skipped_amount else ''
            bar.suffix = bar.suffix.ljust(7) + downloaded + skipped
            bar.update()
            bar.finish()
        if downloaded_counter:
            logger.info(f"Downloaded {downloaded_counter}/{voice_lines_amount} "
                f"for servant {self.collectionNo} (skipped {self.skipped_amount})"
            )
        asyncio.create_task(update_modified_date(self.path_voices))
        logger.debug(f'Finished updating {self.collectionNo} voices')
