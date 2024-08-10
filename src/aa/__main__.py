from sys import argv
from asyncio import run, CancelledError
from typing import Coroutine

from aa.settings import Colors
from aa.settings import Settings
from .downloader.functions import recheck_voice_lines, print_voice_lines


def recheck(args: list[str]) -> Coroutine:
    return recheck_voice_lines()


async def amain():
    async with Settings():
        assert Settings.session is not None
        args = argv[1:]
        if len(args) == 0:
            print(f"{Colors.FAIL}Exception:{Colors.END} First argument is mandatory")
            return
        match args[0]:
            case 'recheck':
                coroutine = recheck(args[1:])
            case 'voices':
                coroutine = voices(args[1:])
            case _:
                print(f"{Colors.FAIL}Unknown Command:{Colors.END} {args[0]}")
                return
        try:
            await coroutine
        except (KeyboardInterrupt, CancelledError):
            pass


def voices(args: list[str]) -> Coroutine:
    return print_voice_lines()


if __name__ == '__main__':
    run(amain())
