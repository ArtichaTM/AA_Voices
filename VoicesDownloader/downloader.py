import logging
from asyncio import run

from progress.bar import Bar

from classes import Downloader, ServantVoices

def setUpLogger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger('AA_voices_downloader')
    logger.setLevel(level=level)
    logging.basicConfig(
        filename='log.txt',
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logger


async def amain():
    logger = setUpLogger(level = logging.INFO)
    logger.info('Starting')
    downloader =  Downloader(delay=1, timeout=30)
    logger.info('Downloader initialized')

    await downloader.recheckAllVoices(bar=Bar)
    await downloader.recheckAllVoices()
    # servant = ServantVoices(27)
    # await servant.buildVoiceLinesDict(False)
    # await servant.updateVoices(bar=Bar())

def main() -> None:
    logger = logging.getLogger('AA_voices_downloader')
    try:
        run(amain())
    except KeyboardInterrupt:
        logger.info("Exiting via KeyboardInterrupt")
    except:
        logger.exception("Uncaught exception:")
        raise
    finally:
        logger.info('Shutdown')

if __name__ == '__main__':
     main()
