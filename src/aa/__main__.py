from typing import Callable, Coroutine
from sys import argv
import asyncio
import subprocess

from aa.settings import Colors
from aa.settings import Settings
from .downloader.functions import recheck_voice_lines, print_voice_lines, count_voice_lines


def recheck(args: list[str]) -> Coroutine:
    return recheck_voice_lines()


def voices(args: list[str]) -> Coroutine:
    return print_voice_lines()


def count(args: list[str]) -> Coroutine:
    return count_voice_lines()


async def convert_diploma_docx(args: list[str]) -> Coroutine:
    target = '-o documents/diploma.docx'
    sources = ['documents/diploma.md']

    if len(args) > 0:
        target = f"-o {args[0]}"
    if len(args) > 1:
        sources = args[1:]

    subprocess.run([
        'python3.11', '-m', 'md2gost',
        *target.split(),
        *sources
    ], shell=True)


FUNCTIONS: dict[str, Callable[[list[str],], Coroutine]] = {
    'recheck': recheck,
    'voices': voices,
    'count': count,
    'convert_diploma': convert_diploma_docx
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
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass
