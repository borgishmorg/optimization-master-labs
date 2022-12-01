import asyncio
import os
import shutil
import requests
import aiohttp
import aiofiles
from graph import timer


# При помощи AsyncIO реализовать алгоритм асинхронного скачивания фалов
# с интернета. Использовать Google Drive или Yandex Disk
# для демонстрации работы приложения


BATCH_SIZE = 16 * 1024


def download_single_file(url: str, folder: str):
    file_name = url.split('/')[-1]
    with open(f'{folder}/{file_name}', 'wb') as f:
        r = requests.get(url)
        assert r.status_code == 200, r.text
        f.write(r.content)


def download_files(urls: list[str], folder: str):
    for url in urls:
        download_single_file(url, folder)


async def download_single_file_aiohttp(url: str, folder: str):
    file_name = url.split('/')[-1]
    with open(f'{folder}/{file_name}', 'wb') as f:
        async with aiohttp.request('GET', url) as r:
            while batch := (await r.content.read(BATCH_SIZE)):
                f.write(batch)


async def download_files_aiohttp(urls: list[str], folder: str):
    tasks = [
        download_single_file_aiohttp(url, folder)
        for url in urls
    ]

    await asyncio.gather(*tasks)


async def download_single_file_aiofiles(url: str, folder: str):
    file_name = url.split('/')[-1]
    async with aiofiles.open(f'{folder}/{file_name}', 'wb') as f:
        async with aiohttp.request('GET', url) as r:
            while batch := (await r.content.read(BATCH_SIZE)):
                await f.write(batch)


async def download_files_aiofiles(urls: list[str], folder: str):
    tasks = [
        download_single_file_aiofiles(url, folder)
        for url in urls
    ]

    await asyncio.gather(*tasks)


def main():
    urls = [
        'https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz',
        'https://www.python.org/ftp/python/3.9.15/Python-3.9.15.tgz',
        'https://www.python.org/ftp/python/3.8.15/Python-3.8.15.tgz',
        'https://www.python.org/ftp/python/3.7.15/Python-3.7.15.tgz',
    ]

    with timer('Preparation:'):
        folder_base = 'download_base'
        folder_aiohttp = 'download_aiohttp'
        folder_aiofiles = 'download_aiofiles'
        shutil.rmtree(folder_base, ignore_errors=True)
        shutil.rmtree(folder_aiohttp, ignore_errors=True)
        shutil.rmtree(folder_aiofiles, ignore_errors=True)
        os.mkdir(folder_base)
        os.mkdir(folder_aiohttp)
        os.mkdir(folder_aiofiles)

    with timer('Base download:'):
        download_files(urls, folder_base)

    with timer('Aiohttp download:'):
        asyncio.run(download_files_aiohttp(urls, folder_aiohttp))

    with timer('Aiofiles download:'):
        asyncio.run(download_files_aiofiles(urls, folder_aiofiles))


if __name__ == '__main__':
    main()
