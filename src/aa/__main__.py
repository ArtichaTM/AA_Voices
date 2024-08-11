from typing import Callable, Coroutine
from sys import argv
from asyncio import run

from aa.settings import Colors
from aa.settings import Settings
from .downloader.functions import recheck_voice_lines, print_voice_lines, count_voice_lines


def recheck(args: list[str]) -> Coroutine:
    return recheck_voice_lines()


def voices(args: list[str]) -> Coroutine:
    return print_voice_lines()


def count(args: list[str]) -> Coroutine:
    return count_voice_lines()


FUNCTIONS: dict[str, Callable[[list[str],], Coroutine]] = {
    'recheck': recheck,
    'voices': voices,
    'count': count
}


async def amain():
    async with Settings():
        assert Settings.session is not None
        args = argv[1:]
        if len(args) == 0:
            print(f"{Colors.FAIL}Exception:{Colors.END} First argument is mandatory")
            return
        func = FUNCTIONS.get(args[0])
        if func is None:
            print(f"{Colors.FAIL}Unknown Command:{Colors.END} {args[0]}")
            return
        try:
            await func(args[1:])
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    try:
        run(amain())
    except KeyboardInterrupt:
        pass
