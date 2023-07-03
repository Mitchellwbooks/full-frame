import asyncio
from unittest import TestCase
from unittest.mock import AsyncMock


class TestFileRecord( TestCase ):

    def test_sync_xmp_updates(self):
        from core.library.FileRecord import FileRecord

        """ Case 1: Image of cheetah """
        file_record = FileRecord()

        file_record.load_xmp_subject = AsyncMock(
            return_value = [
                'A',
                'B'
            ]
        )

        file_record.load_xmp_inference_subject = AsyncMock(
            return_value = [
                'B',
                'C'
            ]
        )

        file_record.add_user_subjects = AsyncMock()

        ''' Function Call '''
        asyncio.run( file_record.sync_xmp_updates() )

        ''' Assertions '''
        file_record.add_user_subjects.assert_awaited_once_with( [ 'A' ] )
