import logging
from asyncio import run
from datetime import datetime
from pathlib import Path

from progress.bar import Bar
from progress.spinner import Spinner

from classes import Downloader

def setUpLogger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger('AA_voices_downloader')
    logger.setLevel(level=level)
    now = datetime.now()
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        filename=f'logs/{now.strftime('%y-%m-%d_%H-%M-%S')}.log',
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logger


async def amain():
    logger = setUpLogger(level = logging.INFO)
    logger.info('Starting')
    downloader =  Downloader(delay=1, maximum_retries=7)
    logger.info('Downloader initialized')

    await downloader.recheckAllVoices(bar=Bar, spinner=Spinner)
    # await downloader.recheckAllVoices()
    # await downloader._print_all_conflicts()

    # from classes import ServantVoices
    # servant = await ServantVoices.load()
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
