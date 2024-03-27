from asyncio import run
from pathlib import Path

from classes import Downloader, ServantVoices

async def main():
    downloader =  Downloader(1)
    await downloader.updateInfo()
    await ServantVoices.load(324)
    await downloader.destroy()


run(main())
