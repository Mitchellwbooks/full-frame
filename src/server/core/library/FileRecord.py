import asyncio
import os
import hashlib
from typing import Dict, List, Literal

from PIL import Image
import rawpy
from aiofile import async_open
from libxmp import consts, XMPMeta
from libxmp.consts import XMP_OPEN_FORUPDATE
from libxmp.exempi import files_get_xmp, files_open_new
from libxmp.utils import file_to_dict

from core.library.Constants import FULL_FRAME_NS_PREFIX, FULL_FRAME_NS_URL, FULL_FRAME_SUBJECT_INFERENCE_LABEL


class FileRecord:
    raw_file_path: str = ''
    raw_file_hash: str = ''
    xmp_file_path: str = None
    xmp_file_hash: str = None
    inferences: List[ Dict ]

    file_type: Literal['raw', 'lossy']

    @classmethod
    async def init(cls, file_path: str, file_extension: str) -> "FileRecord":
        record = cls()
        record.raw_file_path = file_path
        record.xmp_file_path = file_path + '.xmp'

        await record.create_xmp()
        await asyncio.gather(
            record.hash_picture(),
            record.hash_xmp_file()
        )

        raw_files = [
            '.3fr',
            '.ari',
            '.arw',
            '.bay',
            '.braw',
            '.crw',
            '.cr2',
            '.cr3',
            '.cap',
            '.data',
            '.dcs',
            '.dcr',
            '.dng',
            '.drf',
            '.eip',
            '.erf',
            '.fff',
            '.gpr',
            '.iiq',
            '.k25',
            '.kdc',
            '.mdc',
            '.mef',
            '.mos',
            '.mrw',
            '.nef',
            '.nrw',
            '.obm',
            '.orf',
            '.pef',
            '.ptx',
            '.pxn',
            '.r3d',
            '.raf',
            '.raw',
            '.rwl',
            '.rw2',
            '.rwz',
            '.sr2',
            '.srf',
            '.srw',
            '.tif',
            '.x3f'
        ]

        if file_extension in raw_files:
            record.file_type = 'raw'
        else:
            record.file_type = 'lossy'

        return record

    @classmethod
    def from_dict(cls, data: dict ) -> "FileRecord":
        record = cls()
        record.raw_file_path = data["raw_file_path"]
        record.xmp_file_path = data["xmp_file_path"]

        return record

    def __eq__(self, other):
        return all([
            self.raw_file_path == other.raw_file_path,
            self.raw_file_hash == other.raw_file_hash,
            self.xmp_file_path == other.xmp_file_path,
            self.xmp_file_hash == other.xmp_file_hash,
        ])

    def load_pil_image(self) -> "Image":
        if self.file_type == 'raw':
            raw = rawpy.imread(self.raw_file_path)
            rgb = raw.postprocess(use_camera_wb=True)
            pil_image = Image.fromarray(rgb)
        else:
            pil_image = Image.open( self.raw_file_path )

        return pil_image

    async def read_picture(self):
        async with async_open(self.raw_file_path, 'rb') as afp:
            return await afp.read()

    async def create_xmp(self):
        if os.path.isfile(self.xmp_file_path) is False:
            with open(self.xmp_file_path, 'w') as fptr:
                xmp = XMPMeta()
                xmp.parse_from_str('')
                fptr.write(xmp.serialize_to_str())

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

    async def to_dict( self ) -> dict:
        return {
            "raw_file_path": self.raw_file_path,
            "raw_file_hash": await self.hash_picture(),
            "xmp_file_path": self.xmp_file_path,
            "xmp_file_hash": await self.hash_xmp_file(),
            "xmp_subject": await self.load_xmp_subject(),
            "subject_inference": await self.load_xmp_inference_subject()
        }

    async def load_xmp_subject(self) -> List[ str ]:
        with open(self.xmp_file_path, 'r') as fptr:
            xmp = XMPMeta()
            xmp.parse_from_str( fptr.read() )

        return xmp.get_property( consts.XMP_NS_DC, 'subject' )

    async def load_xmp_inference_subject(self) -> List[ str ]:
        with open(self.xmp_file_path, 'r') as fptr:
            xmp = XMPMeta()
            xmp.parse_from_str( fptr.read() )

        return xmp.get_property( FULL_FRAME_NS_URL, FULL_FRAME_SUBJECT_INFERENCE_LABEL )

    def add_label_inferences(self, inferences, model_metadata):
        with open(self.xmp_file_path, 'r+') as fptr:
            strbuffer = fptr.read()
            xmp = XMPMeta()
            xmp.parse_from_str(strbuffer)

            for inference in inferences:
                if inference['confidence'] > .8:
                    xmp.append_array_item(
                        consts.XMP_NS_DC,
                        'subject',
                        inference['label'],
                        {
                            'prop_array_is_ordered': True,
                            'prop_value_is_array': True
                        }
                    )
                    xmp.register_namespace( FULL_FRAME_NS_URL, FULL_FRAME_NS_PREFIX )
                    xmp.append_array_item(
                        FULL_FRAME_NS_URL,
                        FULL_FRAME_SUBJECT_INFERENCE_LABEL,
                        inference['label'],
                        {
                            'prop_array_is_ordered': True,
                            'prop_value_is_array': True
                        }
                    )
            fptr.seek( 0 )
            fptr.write( xmp.serialize_to_str() )
