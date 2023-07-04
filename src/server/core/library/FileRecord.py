import asyncio
import os
import hashlib
import pathlib
from typing import Dict, List, Literal, Tuple

import PIL
from PIL import Image
import rawpy
from aiofile import async_open
from libxmp import consts, XMPMeta

from core.library.Config import Config
from core.library.Constants import FULL_FRAME_NS_PREFIX, FULL_FRAME_NS_URL, KNOWN_SUBJECTS, CURRENT_SUBJECT, \
    PENDING_INFERENCES_SUBJECT, USER_CREATED_SUBJECT, CONFIRMED_INFERENCES_SUBJECT, INCORRECT_INFERENCES_SUBJECT, \
    SUBJECT_PENDING_USER_CONFIRMATION


class FileRecord:
    raw_file_path: str = ''
    raw_file_hash: str = ''
    xmp_file_path: str = None
    xmp_file_hash: str = None
    thumbnail_path: str = ''
    inferences: List[ Dict ]

    file_type: Literal['raw', 'lossy']

    config = Config()

    @classmethod
    async def init( cls, file_path: str ) -> "FileRecord":
        record = cls()
        record.raw_file_path = file_path

        path_hash = hashlib.sha256()
        path_hash.update( file_path.encode( 'utf-8' ) )
        record.thumbnail_path = f'{cls.config.thumbnail_path}/{path_hash.hexdigest()}.jpg'
        record.xmp_file_path = file_path + '.xmp'

        file_extension = pathlib.Path(file_path).suffix

        await record.create_xmp()

        with open(record.xmp_file_path, 'r+') as fptr:
            xmp = XMPMeta()
            xmp.parse_from_str(fptr.read())

            await record.sync_xmp_updates( xmp )

            fptr.seek( 0 )
            fptr.write( xmp.serialize_to_str() )

        if file_extension in cls.config.raw_file_extensions:
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

    async def load_pil_image(self) -> "Image":
        pil_image = await self.create_thumbnail()
        if pil_image is None:
            pil_image = Image.open( self.thumbnail_path )
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
            "CURRENT_SUBJECT": await self.load_xmp_subject( CURRENT_SUBJECT ),
            "PENDING_INFERENCES_SUBJECT": await self.load_xmp_subject( PENDING_INFERENCES_SUBJECT ),
            "CONFIRMED_INFERENCES_SUBJECT": await self.load_xmp_subject( CONFIRMED_INFERENCES_SUBJECT ),
            "INCORRECT_INFERENCES_SUBJECT": await self.load_xmp_subject( INCORRECT_INFERENCES_SUBJECT ),
            "USER_CREATED_SUBJECT": await self.load_xmp_subject( USER_CREATED_SUBJECT ),
        }

    async def load_xmp_subject(self, subject_type: Tuple[ str, str ], xmp: XMPMeta = None ) -> List[ str ]:
        if subject_type not in KNOWN_SUBJECTS:
            raise RuntimeError( 'Subject provided is not known. See Constants.py' )

        if xmp is None:
            with open(self.xmp_file_path, 'r') as fptr:
                xmp = XMPMeta()
                xmp.parse_from_str( fptr.read() )

        labels = []
        label_count = xmp.count_array_items( subject_type[0], subject_type[1] )
        if label_count == 0:
            return []

        for index in range( 1, label_count + 1 ):
            label = xmp.get_array_item( subject_type[0], subject_type[1], index )
            labels.append( label )

        return labels

    async def add_label_inferences(self, inferences, model_metadata):
        with open(self.xmp_file_path, 'r+') as fptr:
            strbuffer = fptr.read()
            xmp = XMPMeta()
            xmp.parse_from_str(strbuffer)

            # :TODO: handle inferences that already exist or have been handled before.
            await self.add_subjects( xmp, CURRENT_SUBJECT, inferences )
            await self.add_subjects( xmp, CURRENT_SUBJECT, [ SUBJECT_PENDING_USER_CONFIRMATION ] )
            await self.add_subjects( xmp, PENDING_INFERENCES_SUBJECT, inferences )

            fptr.seek( 0 )
            fptr.write( xmp.serialize_to_str() )

    async def create_thumbnail(self):
        """
        Creates a thumbnail. Returns image if it didn't exist.
        If one exists, returns None
        """
        if os.path.isfile( self.thumbnail_path ) is False:
            if self.file_type == 'raw':
                raw = rawpy.imread(self.raw_file_path)
                rgb = raw.postprocess(use_camera_wb=True)
                pil_image = Image.fromarray(rgb)
            else:
                pil_image = Image.open(self.raw_file_path)

            pil_image = pil_image.convert( 'RGB' )
            resized_image = pil_image.resize( (224, 224), PIL.Image.LANCZOS )

            resized_image.save( self.thumbnail_path )
            return resized_image
        else:
            return None

    async def add_subjects(self, xmp: XMPMeta, subject_type: Tuple[str, str], subjects: List[ str ] ):
        xmp.register_namespace(FULL_FRAME_NS_URL, FULL_FRAME_NS_PREFIX)

        existing_subjects = await self.load_xmp_subject( subject_type, xmp = xmp )
        for subject in subjects:
            if subject in existing_subjects:
                continue

            xmp.append_array_item(
                subject_type[0],
                subject_type[1],
                subject,
                {
                    'prop_array_is_ordered': True,
                    'prop_value_is_array': True
                }
            )

    async def remove_subjects(self, xmp: XMPMeta, subject_type: Tuple[str, str], subjects: List[ str ] ):
        xmp.register_namespace(FULL_FRAME_NS_URL, FULL_FRAME_NS_PREFIX)

        existing_subjects = await self.load_xmp_subject( subject_type, xmp = xmp )
        subjects_to_keep = set( existing_subjects ) - set( subjects )

        xmp.delete_property( subject_type[0], subject_type[1] )
        for subject in subjects_to_keep:
            if subject in existing_subjects:
                continue

            xmp.append_array_item(
                subject_type[0],
                subject_type[1],
                subject,
                {
                    'prop_array_is_ordered': True,
                    'prop_value_is_array': True
                }
            )
        pass

    async def sync_xmp_updates(self, xmp):
        """
        This function will take the current xmp "subject" field and diff it against various metadata fields.
        This will be used to determine things the user has confirmed vs inferences.

        Let's digress into the set math for these labels:
            "user subjects" ∉ "inference subjects"
            "incorrect inferences" ∉ "current subjects"
            "user subjects" ⋃ "pending inferences" = "current subjects"

        Definitions:
            "current subjects"     = subjects currently attached to the photo; visible in photo editing software.
            "user subjects"        = subjects the user has added on the photo
            "confirmed inferences" = subjects the user has confirmed that were inferenced
            "incorrect inferences" = subjects the user has removed from the photo; indicating it is incorrect
            "pending inferences"   = subjects the user needs to approve
        """
        current_subjects = await self.load_xmp_subject( CURRENT_SUBJECT, xmp = xmp )
        current_cleaned_subjects = set( current_subjects ) - { SUBJECT_PENDING_USER_CONFIRMATION }
        pending_inferences_subject = await self.load_xmp_subject( PENDING_INFERENCES_SUBJECT, xmp = xmp )
        user_subjects = await self.load_xmp_subject( USER_CREATED_SUBJECT, xmp = xmp )
        confirmed_inferences_subject = await self.load_xmp_subject( CONFIRMED_INFERENCES_SUBJECT, xmp = xmp )
        incorrect_inferences_subject = await self.load_xmp_subject( INCORRECT_INFERENCES_SUBJECT, xmp = xmp )

        # User added a subject that is not in our inferences
        user_subjects = \
            current_cleaned_subjects - (
                set( pending_inferences_subject ) |
                set( confirmed_inferences_subject ) |
                set( incorrect_inferences_subject )
            )

        await self.add_subjects( xmp, USER_CREATED_SUBJECT,  list( user_subjects ) )

        if SUBJECT_PENDING_USER_CONFIRMATION in current_subjects:
            return

        # User confirmed our inferences
        confirmed_inferences_to_add = \
            set( pending_inferences_subject ) & set( current_subjects )

        await self.add_subjects( xmp, CONFIRMED_INFERENCES_SUBJECT, list( confirmed_inferences_to_add ) )

        # User removed inference
        incorrect_inferences_to_add = \
            set( pending_inferences_subject ) - set( current_subjects )

        await self.add_subjects( xmp, INCORRECT_INFERENCES_SUBJECT, list( incorrect_inferences_to_add ) )

        # Finish move action
        pending_to_remove = confirmed_inferences_to_add | incorrect_inferences_to_add
        await self.remove_subjects( xmp, PENDING_INFERENCES_SUBJECT, sorted( list( pending_to_remove ) ) )
