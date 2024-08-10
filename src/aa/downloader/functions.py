import asyncio
import os
from time import time

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
    pass
