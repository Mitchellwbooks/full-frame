import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, call

from libxmp import XMPMeta

from core.library.Constants import USER_CREATED_SUBJECT, INCORRECT_INFERENCES_SUBJECT, CONFIRMED_INFERENCES_SUBJECT, \
    PENDING_INFERENCES_SUBJECT, FULL_FRAME_NS_URL, FULL_FRAME_NS_PREFIX, CURRENT_SUBJECT, \
    SUBJECT_PENDING_USER_CONFIRMATION


class TestFileRecord( IsolatedAsyncioTestCase ):

    async def test_sync_xmp_updates(self):
        from core.library.FileRecord import FileRecord

        """
            Case 1: 
                - User added subject 'A'
                - Inference 'C' was removed
                - User marked picture as confirmed (removed SUBJECT_PENDING_USER_CONFIRMATION)
        """
        file_record = FileRecord()

        xmp = XMPMeta()
        xmp.parse_from_str('')
        xmp.register_namespace(FULL_FRAME_NS_URL, FULL_FRAME_NS_PREFIX)

        # CURRENT_SUBJECT
        current_subjects = [
            # Copy to USER_CREATED_SUBJECT
            'A',

            'B'
        ]

        # PENDING_INFERENCES_SUBJECT
        pending_inferences_subjects = [
            # Move to CONFIRMED_INFERENCES_SUBJECT
            'B',
            # Move to INCORRECT_INFERENCES_SUBJECT
            'C'
        ]

        await file_record.add_subjects( xmp, CURRENT_SUBJECT, current_subjects )
        await file_record.add_subjects( xmp, PENDING_INFERENCES_SUBJECT, pending_inferences_subjects )

        file_record.add_subjects = AsyncMock( side_effect = file_record.add_subjects )
        file_record.remove_subjects = AsyncMock( side_effect = file_record.remove_subjects )

        ''' Function Call '''
        await file_record.sync_xmp_updates( xmp )

        ''' Assertions '''
        calls = [
            call( xmp, USER_CREATED_SUBJECT, ['A'] ),
            call( xmp, CONFIRMED_INFERENCES_SUBJECT, ['B'] ),
            call( xmp, INCORRECT_INFERENCES_SUBJECT, ['C'] )
        ]

        file_record.add_subjects.assert_has_calls( calls )

        file_record.remove_subjects.assert_awaited_with(xmp, PENDING_INFERENCES_SUBJECT, ['B', 'C'])

        # Assert final Structure
        incorrect_inferences = await file_record.load_xmp_subject( INCORRECT_INFERENCES_SUBJECT, xmp )
        expected_incorrect_inferences = [
            'C'
        ]
        self.assertEqual( expected_incorrect_inferences, incorrect_inferences )

        confirmed_inferences = await file_record.load_xmp_subject( CONFIRMED_INFERENCES_SUBJECT, xmp )
        expected_confirmed_inferences = [
            'B'
        ]
        self.assertEqual( expected_confirmed_inferences, confirmed_inferences )

        user_subjects = await file_record.load_xmp_subject( USER_CREATED_SUBJECT, xmp )
        expected_user_subjects = [
            'A'
        ]
        self.assertEqual( expected_user_subjects, user_subjects )

        pending_infererences_subject = await file_record.load_xmp_subject( PENDING_INFERENCES_SUBJECT, xmp )
        expected_pending_infererences_subject = [
        ]
        self.assertEqual( expected_pending_infererences_subject, pending_infererences_subject )

        """
            Case 2: 
                - User added subject 'A'
                - Inference 'C' was removed
                - User did not mark picture as confirmed (via removed SUBJECT_PENDING_USER_CONFIRMATION)
        """
        file_record = FileRecord()

        xmp = XMPMeta()
        xmp.parse_from_str('')
        xmp.register_namespace(FULL_FRAME_NS_URL, FULL_FRAME_NS_PREFIX)

        # CURRENT_SUBJECT
        current_subjects = [
            # Copy to USER_CREATED_SUBJECT
            'A',
            'B',
            SUBJECT_PENDING_USER_CONFIRMATION
        ]

        # PENDING_INFERENCES_SUBJECT
        pending_inferences_subjects = [
            'B',
            'C'
        ]

        await file_record.add_subjects( xmp, CURRENT_SUBJECT, current_subjects )
        await file_record.add_subjects( xmp, PENDING_INFERENCES_SUBJECT, pending_inferences_subjects )

        file_record.add_subjects = AsyncMock( side_effect = file_record.add_subjects )
        file_record.remove_subjects = AsyncMock( side_effect = file_record.remove_subjects )

        ''' Function Call '''
        await file_record.sync_xmp_updates( xmp )

        ''' Assertions '''
        calls = [
            call( xmp, USER_CREATED_SUBJECT, ['A'] )
        ]
        file_record.add_subjects.assert_has_calls( calls )

        # Assert final Structure
        pending_inferences = await file_record.load_xmp_subject( PENDING_INFERENCES_SUBJECT, xmp )
        expected_pending_inferences = [
            'B',
            'C'
        ]
        self.assertEqual( expected_pending_inferences, pending_inferences )

        user_subjects = await file_record.load_xmp_subject( USER_CREATED_SUBJECT, xmp )
        expected_user_subjects = [
            'A'
        ]
        self.assertEqual( expected_user_subjects, user_subjects )
