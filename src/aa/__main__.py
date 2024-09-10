from typing import Callable, Coroutine
from sys import argv
import asyncio
import subprocess

from aa.settings import Colors
from aa.settings import Settings
from .downloader.functions import (
    recheck_voice_lines,
    print_voice_lines,
    count_voice_lines,
    clean_unused_data
)


def recheck(args: list[str]) -> Coroutine:
    """
    Verifies loaded voice lines
    and loads the missing ones
    """
    return recheck_voice_lines()


def voices(args: list[str]) -> Coroutine:
    """
    Prints path and subtitle for each valid voice line
    """
    return print_voice_lines()


def count(args: list[str]) -> Coroutine:
    """ Counts valid, invalid voice lines and servants """
    return count_voice_lines()


def cleanup(args: list[str]) -> Coroutine:
    """ Cleans servant folders with leftovers or other files """
    return clean_unused_data()


async def help(args: list[str]) -> None:
    """
    Prints help docstring from function. Example:
    > help help
    """
    if not args:
        args = ['help']

    func = FUNCTIONS.get(args[0])
    if func is None:
        print(f"{Colors.FAIL}Unknown Command{Colors.END}: {Colors.BOLD}{args[0]}{Colors.END}")
        return
    if func.__doc__ is None:
        print(f"Command {Colors.BOLD}{args[0]}{Colors.END} {Colors.FAIL}has no{Colors.END} help string)")
    print(func.__doc__)


async def convert_diploma_docx(args: list[str]) -> None:
    """
    Converts diploma that resides in data/documents
    from markdown to docx
    Arg1 : target where to place converted file
    Arg2+: sources to convert from, separated by whitespace
    """
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
    'convert_diploma': convert_diploma_docx,
    'cleanup': cleanup,
    'help': help
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
            print(f"{Colors.FAIL}Unknown Command{Colors.END}: {Colors.BOLD}{args[0]}{Colors.END}")
            print(f"{Colors.GREEN}Available{Colors.END}: {Colors.BOLD}", end='')
            print(f'{Colors.END}, {Colors.BOLD}'.join(FUNCTIONS.keys()) + Colors.END)
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
