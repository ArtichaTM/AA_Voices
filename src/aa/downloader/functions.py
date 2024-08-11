import asyncio
import os
from time import time
from pathlib import Path

from aiopath import AsyncPath

from aa.settings import Settings
from .basic_servant import BasicServants
from .servant import Servant
from .classes import ThrottleLastProgressBar

__all__ = ('recheck_voice_lines', )


async def update_modified_date(path: AsyncPath):
    """ Set's path modify time to now
    :param path: Target path
    """
    assert await path.exists()
    stats = await path.lstat()
    os.utime(str(path), (stats.st_ctime, time()))


async def get_timestamp() -> int:
    async with Settings.i.get('/info') as resp:
        info = await resp.json()
        assert isinstance(info, dict)
        assert 'NA' in info
        info = info['NA']
        assert isinstance(info, dict)
        assert 'timestamp' in info
        timestamp = info['timestamp']
        assert isinstance(timestamp, int)
        return timestamp


async def recheck_voice_lines():
    latest_aa_update = await get_timestamp()
    svts = await BasicServants.load()
    for basic_servant in svts.iter():
        progress = ThrottleLastProgressBar()
        await progress.set_text("Downloading info from AA")
        svt_local = Servant.fromBasicServant(basic_servant)
        local_json_exists = await svt_local.json_path.exists()

        if not local_json_exists:
            svt_aa = Servant.fromBasicServant(basic_servant)
            await svt_aa.load_from_aa()
            svt_aa.full_parse()
            asyncio.create_task(svt_aa.save_json())
            await svt_aa.update_from(svt_aa, progress=progress)
        else:
            stat = await svt_local.json_path.stat()
            await svt_local.load_from_json()
            svt_local.full_parse()
            svt_aa = Servant.fromBasicServant(basic_servant)
            if stat.st_mtime > latest_aa_update:
                svt_aa = svt_local
            else:
                await svt_aa.load_from_aa()
                svt_aa.full_parse()
                asyncio.create_task(svt_aa.save_json())
            await svt_local.update_from(svt_aa, progress=progress)
        await progress.finish()


async def print_voice_lines() -> None:
    assert Settings.i is not None
    svts_path = Settings.i.servants_path
    svt_max = max([int(i.name) async for i in svts_path.iterdir()])
    for collectionNo in range(1, svt_max):
        try:
            servant = await Servant.fromCollectionNo(collectionNo).load_from_json()
        except FileNotFoundError():
            continue
        servant.full_parse()
        for path, string in servant.voices.iter_subtitles():
            path = Path(path).resolve()
            print(path, string.replace('\n', ' '))


async def _count_voice_lines_servant(collectionNo: int) -> tuple[int, int] | None:
    """Returns amount of valid and invalid voices lines for servant"""
    assert isinstance(collectionNo, int)
    valid = 0
    invalid = 0
    try:
        servant = await Servant.fromCollectionNo(collectionNo).load_from_json()
    except FileNotFoundError():
        return None
    servant.full_parse()
    for voice_line in servant.voices.iter_voice_lines():
        for path in voice_line.voice_lines_paths():
            if await path.exists():
                valid += 1
            else:
                invalid += 1
    return valid, invalid


def _count_voice_lines_format(valid: int, invalid: int, svts: int) -> str:
    print(f"\r{valid: >5} | {invalid: >7} | {svts: >17}", end='', flush=True)


async def count_voice_lines() -> None:
    assert Settings.i is not None
    svts_path = Settings.i.servants_path

    tasks = []
    async for svt_path in svts_path.iterdir():
        try:
            collectionNo = int(svt_path.name)
        except ValueError:
            continue
        tasks.append(asyncio.create_task(_count_voice_lines_servant(collectionNo)))

    valid = 0
    invalid = 0
    svt_counter = 0
    print(f"Valid | Invalid | Servants Processed")
    _count_voice_lines_format(valid, invalid, svt_counter)
    for completed in asyncio.as_completed(tasks):
        counters = await completed
        if counters is not None:
            valid += counters[0]
            invalid += counters[1]
        svt_counter += 1
        _count_voice_lines_format(valid, invalid, svt_counter)
    _count_voice_lines_format(valid, invalid, svt_counter)
