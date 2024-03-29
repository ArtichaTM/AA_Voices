from asyncio import run

from classes import Downloader, ServantVoices

async def main():
    downloader =  Downloader(1)
    try:
        await downloader.updateInfo()
        voices = await ServantVoices.load(324)
        await voices.buildVoiceLinesDict(fill_all_ascensions=False)
        await voices.updateVoices();
    finally:
        await downloader.destroy()

if __name__ == '__main__':
    run(main())
