from typing import NamedTuple
import asyncio
import re
from json import dumps, loads
from typing import Generator
from warnings import warn

from aiopath import AsyncPath

from aa.settings import Settings
from .basic_servant import BasicServant
from .classes import ProgressWatcher


VOICE_LINE_TYPE_TRANSLATIONS: dict[str, str] = {
    'home': 'Home',
    'groeth': 'Growth',
    'battle': 'Battle',
    'masterMission': 'MasterMission',
    'treasureDevice': 'NoblePhantasm',
    'firstGet': 'Summon',
    'eventReward': 'EventReward',
    'boxGachaTalk': 'EventGachaTalk',
    'eventShop': 'EventShopTalk',
    'eventShopPurchase': 'EventShopPurchase',
    'eventJoin': 'JoinEvent',
    'guide': 'Guide',
    'eventTowerReward': 'EventReward',
    'warBoard': 'WarBoard',
    'eventDailyPoint': 'EventDailyPoint',
    'treasureBox': 'Treasure',
    'eventDigging': 'EventDigging'
}
RESTRICTED_IN_PATH = set(r'"\/{},.!?<>:;' + "'\\\r\n")


def verify_path_applicable(string: str) -> str:
    assert Settings.i is not None
    if Settings.i.replace_spaces:
        RESTRICTED_IN_PATH.add(' ')
    return ''.join(('_' if i in RESTRICTED_IN_PATH else i for i in string)).strip('_ ')


class Subtitle(NamedTuple):
    svt_id: int
    voice_line_id: str
    text: str
    url: str
    path: AsyncPath


class VoiceLine:
    """ Contains Voices['voiceLines'][i] """
    __slots__ = ('json', 'parent')

    def __init__(self, parent: 'VoiceType', voice_line_data: dict) -> None:
        assert isinstance(parent, VoiceType)
        assert isinstance(voice_line_data, dict)
        self.parent = parent
        self.json = voice_line_data

    def __getitem__(self, item):
        return self.json[item]

    def __repr__(self) -> str:
        return f"<VoiceLine {self.parent.parent.svt.collectionNo}/{self.parent['type']}/{self.anyName}>"

    def __eq__(self, value: object) -> bool:
        assert isinstance(value, VoiceLine)
        return tuple(self['id']) == tuple(value['id'])

    def __len__(self) -> int:
        return len(self.json['id'])

    @property
    def folder(self) -> AsyncPath:
        """ Path to the folder containing audio files """
        assert Settings.i is not None
        target_folder = self.parent.path
        folder_name = verify_path_applicable(self.anyName)
        assert isinstance(folder_name, str)
        if Settings.i.replace_spaces:
            folder_name = verify_path_applicable(folder_name)
        target_folder /= folder_name
        return target_folder

    @property
    def anyName(self) -> str:
        """ Returns any voice line name """
        assert 'overwriteName' in self.json
        string = self['overwriteName']
        if string:
            return string
        elif 'name' in self.json and self['name']:
            return self['name']
        else:
            assert 'id' in self.json
            return self.json['id'][0]

    def subtitle_split(self) -> Generator[Subtitle, None, None]:
        """ Iterate over subtitles """
        assert 'subtitle' in self.json
        subtitle = self['subtitle']
        assert isinstance(subtitle, str)
        if subtitle is None:
            yield from ()
        counter = 0
        paths = self.voice_lines_paths()
        for part in re.finditer(r'\[id (\d+)_(\d_[^\]]*)\]([^\[]*)[^\[]', subtitle):
            svt_id, voice_line_id, text = part.groups()
            svt_id = int(svt_id)
            assert len(self['audioAssets']) > counter, self
            yield Subtitle(
                svt_id,
                voice_line_id,
                text.strip(),
                self['audioAssets'][counter],
                next(paths)
            )
            counter += 1

    def voice_lines_paths(self) -> Generator[AsyncPath, None, None]:
        """ Iterate over audio files paths """
        target_folder = self.folder
        for id in self['id']:
            assert isinstance(id, str)
            file_path = target_folder / f"{id}.mp3"
            yield file_path

    def audio_files(self) -> Generator[tuple[AsyncPath, str], None, None]:
        """ Iterate over files paths and their URLs on AA """
        for file_path, url in zip(self.voice_lines_paths(), self['audioAssets']):
            assert isinstance(url, str)
            assert url.startswith('https://')
            yield file_path, url

    async def verify(self, progress: ProgressWatcher) -> None:
        """ Do nothing if audio files exists, else do nothing """
        assert Settings.i is not None
        await self.folder.mkdir(exist_ok=True)
        for file_path, url in self.audio_files():
            exists = await file_path.exists()
            if not exists:
                async with Settings.i.get(url) as resp:
                    await progress.set_text(self.anyName)
                    payload = await resp.read()
                    await file_path.write_bytes(payload)
            await progress.current_add()

    async def download(self, progress: ProgressWatcher) -> None:
        """ Downloads current voice line separated """
        assert Settings.i is not None
        await self.folder.mkdir(exist_ok=True)
        for file_path, url in self.audio_files():
            async with Settings.i.get(url) as resp:
                payload = await resp.read()
                await file_path.write_bytes(payload)
            await progress.current_add()

    async def delete(self, missing_ok: bool = False) -> None:
        """ Removes audio files """
        async with asyncio.TaskGroup() as tg:
            for file_path, _ in self.audio_files():
                tg.create_task(file_path.unlink(missing_ok=missing_ok))

            # Checking, if folder is empty. If empty, than delete folder
            folder = self.folder
            while True:
                finished = False
                async for _ in folder.iterdir():
                    finished = True
                    break
                else:
                    tg.create_task(folder.rmdir())
                    folder = folder.parent
                if finished:
                    break


class VoiceType:
    """ Contains Servant['profile']['voices'][i]"""
    __slots__ = ('parent', 'json', '_voice_lines', 'parsed')

    def __init__(self, parent: 'ServantVoices', voices_data: dict) -> None:
        assert isinstance(parent, ServantVoices)
        self.parent = parent
        self.json = voices_data
        self._voice_lines: dict[str, VoiceLine] = dict()
        self.parsed = False

    def __getitem__(self, item):
        return self.json[item]

    def __repr__(self) -> str:
        return f"<Voices {self['svtId']}/{self['voicePrefix']}>"

    def __eq__(self, value: object) -> bool:
        assert isinstance(value, VoiceType)
        if len(self.voice_lines) != len(value.voice_lines):
            return False
        for key in {'svtId', 'voicePrefix', 'type'}:
            if self[key] != value[key]:
                return False
        for left, right in zip(self.voice_lines, value.voice_lines):
            if left != right:
                return False
        return True

    def __hash__(self) -> int:
        assert 'svtId' in self.json
        assert 'voicePrefix' in self.json
        assert 'type' in self.json
        return hash((
            self.json['svtId'],
            self.json['voicePrefix'],
            self.json['type']
        ))

    def __len__(self) -> int:
        return sum(len(i) for i in self.voice_lines.values())

    @property
    def path(self) -> AsyncPath:
        assert 'svtId' in self.json
        assert 'voicePrefix' in self.json
        assert 'type' in self.json
        assert self['type'] in VOICE_LINE_TYPE_TRANSLATIONS, self['type'] + ' not found in translations'
        target: AsyncPath = self.parent.svt.voices_folder
        target /= f"{self['svtId']}_{self['voicePrefix']}"
        target /= VOICE_LINE_TYPE_TRANSLATIONS[self['type']]
        return target

    @property
    def voice_lines(self):
        if not self.parsed:
            warn("Parsing Voices.voice_lines during request. Preload first")
            self.parse()
        return self._voice_lines

    def parse(self) -> None:
        if self.parsed:
            warn("Repeat Voices.parse() call")
            return
        for line in self['voiceLines']:
            vl = VoiceLine(self, line)
            assert 'id' in vl.json
            assert isinstance(vl['id'], list)
            assert len(vl['id']) > 0
            first_id = vl['id'][0]
            assert isinstance(first_id, str)
            self._voice_lines[first_id] = vl
        self.parsed = True

    async def download(self, progress: ProgressWatcher) -> None:
        """ Complete download. Erases previous folder, if exists """
        await self.delete(missing_ok=True)
        await self.path.mkdir(parents=True)
        async with asyncio.TaskGroup() as tg:
            for voice_line in self.voice_lines.values():
                tg.create_task(voice_line.download(progress=progress))

    async def update_from(self, other: 'VoiceType', progress: ProgressWatcher) -> None:
        self_keys = set(self.voice_lines.keys())
        other_keys = set(self.voice_lines.keys())
        intersection_keys = self_keys & other_keys
        await self.path.mkdir(parents=True, exist_ok=True)
        async with asyncio.TaskGroup() as tg:
            for common_key in intersection_keys:
                if self.voice_lines[common_key] == other.voice_lines[common_key]:
                    tg.create_task(other.voice_lines[common_key].verify(progress=progress))
                else:
                    tg.create_task(self.voice_lines[common_key].download(progress=progress))
            # Returning if they are no new costumes or voice line types
            if len(intersection_keys) == len(self_keys):
                return

            # Exists in old, but not in new (Something removed)
            removed_keys = self_keys - other_keys
            for removed_key in removed_keys:
                tg.create_task(self.voice_lines[removed_key].delete())

            # Exists in new, but not in old (Something added)
            added_keys = other_keys - self_keys
            for added_key in added_keys:
                tg.create_task(self.voice_lines[added_key].download(progress=progress))

    async def delete(self, missing_ok: bool = False) -> None:
        """ Removes downloaded folders and audio files """
        for voice_line in self.voice_lines.values():
            await voice_line.delete(missing_ok=missing_ok)


class CostumeInfo:
    """ Contains Servant['profile']['costume'][i]"""
    __slots__ = ('json',)

    def __init__(self, costume_data: dict) -> None:
        self.json = costume_data

    def __getitem__(self, item):
        return self.json[item]

    def __eq__(self, value: object) -> bool:
        assert isinstance(value, CostumeInfo)
        for key in {'id', 'costumeCollectionNo'}:
            if self[key] != value[key]:
                return False
        return True


class ServantVoices:
    """ Contains CostumeInfo and Voices """
    __slots__ = ('svt', '_costumes', '_voices', 'parsed')

    def __init__(self, svt: 'Servant') -> None:
        self.svt = svt
        self.parsed = False
        self._costumes: dict[int, CostumeInfo] = dict()
        self._voices: dict[tuple[int, int, str], VoiceType] = dict()

    def __eq__(self, value: object) -> bool:
        assert isinstance(value, ServantVoices)
        if self.costumes != value.costumes:
            return False
        if self.voices != value.voices:
            return False
        return True

    def __len__(self) -> int:
        return sum(len(i) for i in self.voices.values())

    @property
    def costumes(self):
        if not self.parsed:
            warn("Parsing ServantVoices.costumes during request. Preload first")
            self.parse()
        return self._costumes

    @property
    def voices(self):
        if not self.parsed:
            warn("Parsing ServantVoices.voices during request. Preload first")
            self.parse()
        return self._voices

    def parse(self) -> None:
        assert self.svt.json_loaded, f"Servant should be loaded to parse voices"
        if self.parsed:
            warn("Repeat ServantVoices.parse() call")
            return
        current = self.svt.json
        assert isinstance(current, dict)
        assert 'profile' in current, current.keys()
        current = current['profile']
        assert isinstance(current, dict)
        assert 'costume' in current, current.keys()
        assert 'voices' in current, current.keys()
        costumes: dict[str, dict] = current['costume']
        voices: list[dict] = current['voices']
        assert isinstance(costumes, dict)
        assert isinstance(voices, list)
        for index, data in costumes.items():
            assert isinstance(index, str), index
            assert isinstance(data, dict), data
            self._costumes[int(index)] = CostumeInfo(data)
        for data in voices:
            assert isinstance(data, dict)
            v = VoiceType(self, data)
            self._voices[(
                v['svtId'],
                v['voicePrefix'],
                v['type']
            )] = v
        self.parsed = True

    def full_preload(self) -> None:
        self.parse()
        for voices in self.voices.values():
            voices.parse()

    def iter_voices(self) -> Generator[VoiceType, None, None]:
        for voice in self.voices.values():
            yield voice

    def iter_voice_lines(self) -> Generator[VoiceLine, None, None]:
        for voice in self.iter_voices():
            for voice_line in voice.voice_lines.values():
                yield voice_line

    def iter_subtitles(self) -> Generator[tuple[AsyncPath, str], None, None]:
        for voice_line in self.iter_voice_lines():
            for subtitle in voice_line.subtitle_split():
                yield subtitle.path, subtitle.text

    async def update_from(self, other: 'ServantVoices', progress: ProgressWatcher) -> None:
        self_keys = set(self.voices.keys())
        other_keys = set(self.voices.keys())
        intersection_keys = self_keys & other_keys
        async with asyncio.TaskGroup() as tg:
            for common_key in intersection_keys:
                tg.create_task(
                    self.voices[common_key].update_from(
                        other.voices[common_key],
                        progress=progress
                    )
                )
            # Returning if they are no new costumes or voice line types
            if len(intersection_keys) == len(self_keys):
                return

            # Exists in old, but not in new (Something removed)
            removed_keys = self_keys - other_keys
            for removed_key in removed_keys:
                tg.create_task(self.voices[removed_key].delete())

            # Exists in new, but not in old (Something added)
            added_keys = other_keys - self_keys
            for added_key in added_keys:
                tg.create_task(self.voices[added_key].download(progress=progress))


class Servant:
    """
    Initialization completes in 2 steps:
    1. Creating class via class methods named "from*"
    2. Loading info via methods named "load_from_*"
    """
    __slots__ = (
        'collectionNo',
        'json_loaded', 'json',
        'voices'
    )

    def __init__(self, collectionNo: int) -> None:
        assert isinstance(collectionNo, int)
        self.collectionNo = collectionNo
        self.json: dict | None = None
        self.json_loaded: bool = False
        self.voices: ServantVoices = ServantVoices(self)

    def __getitem__(self, item):
        assert self.json is not None
        return self.json[item]

    def __eq__(self, value: object) -> bool:
        assert isinstance(value, Servant)
        assert self.json_loaded, f"Servant should be loaded to compare it"
        if self.collectionNo != value.collectionNo:
            return False
        return self.voices == value.voices

    @property
    def root_folder(self) -> AsyncPath:
        assert Settings.i is not None
        return Settings.i.servants_path / str(self.collectionNo)

    @property
    def voices_folder(self) -> AsyncPath:
        assert Settings.i is not None
        return self.root_folder / Settings.i.voices_folder_name

    @property
    def json_path(self) -> AsyncPath:
        return self.root_folder / 'data.json'

    @property
    def anyName(self) -> str:
        return self['battleName']

    @classmethod
    def fromCollectionNo(cls, collectionNo: int) -> 'Servant':
        assert isinstance(collectionNo, int)
        return cls(collectionNo)

    @classmethod
    def fromBasicServant(cls, basic: BasicServant) -> 'Servant':
        assert isinstance(basic, BasicServant)
        return cls.fromCollectionNo(basic.collectionNo)

    async def load_from_json(self) -> 'Servant':
        """ Loads json from file system. FileNotFoundError exception, if json does not exist """
        if not await self.json_path.exists():
            raise FileNotFoundError()
        bytes = await self.json_path.read_bytes()
        self.json = loads(bytes)
        self.json_loaded = True
        return self

    async def load_from_aa(self) -> 'Servant':
        """ Loads json directly from atlas academy """
        assert Settings.i is not None
        async with Settings.i.get(f"/nice/NA/servant/{self.collectionNo}?lore=true") as resp:
            self.json = await resp.json()
            assert isinstance(self.json, dict)
            assert 'detail' not in self.json
        self.json_loaded = True
        return self

    async def save_json(self) -> None:
        """ Saves current json to filesystem """
        assert self.json_loaded, f"Servant should be loaded to save json"
        await self.root_folder.mkdir(exist_ok=True)
        text = dumps(self.json)
        await self.json_path.write_text(text)

    def full_parse(self) -> None:
        """ Parses everything possible """
        assert self.json_loaded, f"Servant should be loaded to preload it's data"
        self.voices.full_preload()

    async def download(self, progress: ProgressWatcher) -> None:
        await progress.set_text(f"{self.anyName}")
        await progress.set_maximum(len(self.voices))
        await self.voices.update_from(self.voices, progress=progress)
        await progress.set_text(f"{self.anyName}")
        await progress.finish()

    async def update_from(self, new_svt: 'Servant', progress: ProgressWatcher) -> None:
        await progress.set_text(f"{new_svt.collectionNo: >3}. {new_svt.anyName}")
        await progress.set_maximum(len(self.voices))
        await self.voices.update_from(new_svt.voices, progress=progress)
        await progress.set_text(f"{new_svt.collectionNo: >3}. {new_svt.anyName}")

