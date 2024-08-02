from pathlib import Path
from enum import IntEnum

__all__ = (
    'ExceptionType',
    'ALBUM_NAME',
    'MAINDIR',
    'SERVANT_EXCEPTIONS',
    'API_SERVER',
    'SERVANTS_FOLDER',
    'VOICES_FOLDER_NAME',
    'FFMPEG_PATH'
)


class ExceptionType(IntEnum):
    NP_IN_BATTLE_SECTION = 0
    SKIP_ON_DOWNLOAD_EXCEPTION = 1

ALBUM_NAME = 'Fate: Grand Order Servants'
MAINDIR = Path(__file__).parent.joinpath('..').absolute()
DATASET_DIR = MAINDIR / 'dataset'
LOGS_PATH = MAINDIR / 'logs'
SERVANTS_FOLDER = DATASET_DIR / 'Servants'
VOICES_FOLDER_NAME = 'voices'
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
    , 339: {ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION, }
    , 341: {ExceptionType.SKIP_ON_DOWNLOAD_EXCEPTION, }
}

API_SERVER: str = r'https://api.atlasacademy.io'
FFMPEG_PATH: str = 'ffmpeg'
