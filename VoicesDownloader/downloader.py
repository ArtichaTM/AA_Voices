import logging
from asyncio import run

from progress.bar import Bar

from classes import Downloader

def setUpLogger(level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger('AA_voices_downloader')
    logger.setLevel(level=level)
    logging.basicConfig(
        filename='log.txt',
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logger


async def main():
    logger = setUpLogger(level = logging.INFO)
    logger.info('Starting')
    downloader =  Downloader(delay=1, timeout=30)
    logger.info('Downloader initialized')
    try:
        await downloader.recheckAllVoices(bar=Bar)
    finally:
        await downloader.destroy()
        logger.info('Shutdown')

if __name__ == '__main__':
    run(main())
