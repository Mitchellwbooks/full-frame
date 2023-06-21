import asyncio
import os
import hashlib
from aiofile import async_open


class FileRecord:
    raw_file_path: str = ''
    raw_file_hash: str = ''
    xmp_file_path: str = None
    xmp_file_hash: str = None

    @classmethod
    async def init(cls, file_path: str, file_extension: str) -> "FileRecord":
        record = cls()
        record.raw_file_path = file_path
        record.xmp_file_path = file_path.replace(file_extension, '.xmp')

        await record.create_xmp()
        await asyncio.gather(
            record.hash_picture(),
            record.hash_xmp_file()
        )

    @classmethod
    def from_dict(cls, data: dict ) -> "FileRecord":
        record = cls()
        record.raw_file_path = data["raw_file_path"]
        record.raw_file_hash = data["raw_file_hash"]
        record.xmp_file_path = data["xmp_file_path"]
        record.xmp_file_hash = data["xmp_file_hash"]

        return record

    def __eq__(self, other):
        return all([
            self.raw_file_path == other.raw_file_path,
            self.raw_file_hash == other.raw_file_hash,
            self.xmp_file_path == other.xmp_file_path,
            self.xmp_file_hash == other.xmp_file_hash,
        ])

    async def read_picture(self):
        async with async_open(self.raw_file_path, 'rb') as afp:
            return await afp.read()

    async def create_xmp(self):
        from libxmp import XMPFiles
        if os.path.isfile(self.xmp_file_path) is False:
            xmp_file = XMPFiles(file_path=self.xmp_file_path)
            xmp_file.close_file()

    async def hash_picture(self):
        self.raw_file_hash = await self.file_md5_digest(self.raw_file_path)

    async def hash_xmp_file(self):
        self.xmp_file_hash = await self.file_md5_digest(self.xmp_file_path)

    @staticmethod
    async def file_md5_digest(file_path):
        md5_hash = hashlib.md5()
        async with async_open(file_path, 'rb') as afp:
            async for chunk in afp.iter_chunked():
                md5_hash.update(chunk)

        return md5_hash.hexdigest()

    def to_dict( self ) -> dict:
        return {
            "raw_file_path": self.raw_file_path,
            "raw_file_hash": self.raw_file_hash,
            "xmp_file_path": self.xmp_file_path,
            "xmp_file_hash": self.xmp_file_hash,
        }
