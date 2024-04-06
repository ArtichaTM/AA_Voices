import logging
from asyncio import run

from progress.bar import Bar

from classes import Downloader, ServantVoices

def setUpLogger(level: int = logging.DEBUG) -> None:
    logger = logging.getLogger('AA_voices_downloader')
    logger.setLevel(level=level)


async def main():
    setUpLogger()
    downloader =  Downloader(delay=2, timeout=30)
    try:
        await downloader.updateInfo()
        voices = await ServantVoices.load(121)
        await voices.buildVoiceLinesDict(fill_all_ascensions=False)
        bar = Bar()
        await voices.updateVoices(bar=bar)
    finally:
        await downloader.destroy()

if __name__ == '__main__':
    run(main())
