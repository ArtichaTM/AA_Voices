import logging
from asyncio import run

from progress.bar import Bar

from classes import Downloader, ServantVoices

def setUpLogger(level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger('AA_voices_downloader')
    logger.setLevel(level=level)
    logging.basicConfig(filename='log.txt', level=logging.DEBUG)
    return logger


async def main():
    logger = setUpLogger(level = logging.INFO)
    logger.info('Starting')
    downloader =  Downloader(delay=1, timeout=30)
    logger.info('Downloader initialized')
    try:
        await downloader.updateInfo()
        logger.info('Info updated')
        for i in range(1, 338):
            voices = await ServantVoices.load(i)
            await voices.buildVoiceLinesDict(fill_all_ascensions=False)
            await voices.updateVoices(bar=Bar())
    finally:
        await downloader.destroy()
        logger.info('Shutdown')

if __name__ == '__main__':
    run(main())
