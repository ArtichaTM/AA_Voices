import logging
from asyncio import run

from progress.bar import Bar
from progress.spinner import Spinner

from classes import Downloader

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
    downloader =  Downloader(delay=1, maximum_retries=7)
    logger.info('Downloader initialized')

    await downloader.recheckAllVoices(bar=Bar, _spinner=Spinner)
    # await downloader.recheckAllVoices()
    # await downloader._print_all_conflicts()

    # from classes import ServantVoices
    # servant = await ServantVoices.load(142)
    # servant.buildVoiceLinesDict(False)
    # await servant.updateVoices(bar=Bar())
    # await servant.updateVoices()
    # servant._print_conflicts()

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
