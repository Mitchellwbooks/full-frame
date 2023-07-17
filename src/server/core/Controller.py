import asyncio
import os
from multiprocessing import Process, Queue
from typing import List, Dict, Set

from core.library.Config import Config
from core.library.Constants import CURRENT_SUBJECT
from core.library.FileRecord import FileRecord


class Controller(Process):
    """
    Abstracts User Files from rest of program
    """
    continue_processing = True
    controller_to_model_manager: Queue
    model_manager_to_controller: Queue
    controller_to_inferencer: Queue
    inferencer_to_controller: Queue
    current_graph: List[ "FileRecord" ]
    config: "Config"

    def __init__(
            self,
            controller_to_model_manager,
            model_manager_to_controller,
            controller_to_inferencer,
            inferencer_to_controller
    ):
        super().__init__()
        self.controller_to_model_manager = controller_to_model_manager
        self.model_manager_to_controller = model_manager_to_controller
        self.controller_to_inferencer = controller_to_inferencer
        self.inferencer_to_controller = inferencer_to_controller
        self.config = Config()
        self.current_file_dict = {}
        self.continue_processing = True

    def run(self):
        asyncio.run(self.async_run())

    async def async_run(self):
        await asyncio.gather(
            self.run_read_inferencer_messages(),
            self.run_scan_files()
        )

    async def run_read_inferencer_messages(self):
        while self.continue_processing:
            if self.inferencer_to_controller.qsize() == 0:
                await asyncio.sleep(10)
                continue

            message = self.inferencer_to_controller.get()

            if message['topic'] == 'inference_made':
                # Refresh local file to initialize with changes.
                file_record = message['record']
                file_lookup = hash( file_record )
                local_record = self.current_file_dict[ file_lookup ]
                self.current_file_dict[ file_lookup ] = await FileRecord.init( local_record.raw_file_path )

    async def run_scan_files( self ):
        while self.continue_processing:
            await self.run_full_scan()

            # Re-run a scan every 5 minutes
            await asyncio.sleep( 5 * 60 )

    async def run_full_scan(self):
        print( f'Running Scan on {self.config.folder_path}' )

        new_file_dict: Dict[ int, FileRecord ] = await self.load_files_from_folder( self.config.folder_path )

        new_file_set = { item for item in new_file_dict.values() }
        old_file_set = { item for item in self.current_file_dict.values() }

        removed_files = old_file_set - new_file_set
        discovered_files: Set[ FileRecord ] = new_file_set - old_file_set
        common_files = new_file_set & old_file_set
        files_with_metadata_changes = []

        # Determine files with changes
        for file in common_files:
            old_file = self.current_file_dict[ hash( file ) ]
            new_file = new_file_dict[ hash( file ) ]
            if old_file.xmp_file_hash != new_file.xmp_file_hash:
                files_with_metadata_changes.append( new_file )

        # Send file events along
        print('discovered files:')
        for record in discovered_files:
            print( f'\tFile Path: {record.raw_file_path}')
            print( f'\t\tXMP Subjects{await record.load_xmp_subject( CURRENT_SUBJECT )}')
            event = {
                'topic': 'discovered_file',
                'file_record': record
            }
            self.controller_to_inferencer.put( event )
            self.controller_to_model_manager.put( event )

        for record in removed_files:
            event = {
                'topic': 'removed_file',
                'file_record': record
            }
            self.controller_to_inferencer.put( event )
            self.controller_to_model_manager.put( event )

        for record in files_with_metadata_changes:
            event = {
                'topic': 'metadata_file_changed',
                'file_record': record
            }
            self.controller_to_inferencer.put( event )
            self.controller_to_model_manager.put( event )

        print( f'Discovered Files: {len( discovered_files )}')
        print( f'Removed Files: {len(removed_files)}' )
        print( f'Files With Metadata Changes: {len(files_with_metadata_changes)}' )

        # Save new state
        self.current_file_dict = new_file_dict

    async def load_files_from_folder(self, folder):
        init_coros = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_extension_matches = False
                for extension in self.config.raw_file_extensions:
                    if file.lower().endswith(extension):
                        file_extension_matches = True
                        break

                if file_extension_matches:
                    file_record_coro = FileRecord.init( f'{root}/{file}' )
                    init_coros.append(file_record_coro)

        file_list: List[ FileRecord, Exception ] = await asyncio.gather( *init_coros, return_exceptions = True )

        file_dict = {}
        for index, file in enumerate( file_list ):
            if isinstance( file, Exception ):
                # An error occurred when loading this file object.
                del file_list[ index ]
                print( file )
            else:
                file_dict[ hash( file) ] = file

        return file_dict


