import asyncio
import json
import os
from multiprocessing import Process, Queue
from typing import List

from core.library.Config import Config
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

    def run(self):
        self.continue_processing = True
        asyncio.run(self.async_run())

    async def async_run(self):
        await self.load_current_filesystem_graph()
        await self.run_full_scan()
        while self.continue_processing:
            await asyncio.sleep( 10 )
            pass

    async def run_full_scan(self):
        init_coros = []
        for root, dirs, files in os.walk(self.config.folder_path):
            for file in files:
                for extension in self.config.file_extensions:
                    if file.endswith( extension ):
                        file_record_coro = FileRecord.init( f'{root}/{file}', extension )
                        init_coros.append( file_record_coro )
                        break

        new_graph = await asyncio.gather( *init_coros )

        new_graph_set = { item for item in new_graph }
        old_graph_set = { item for item in self.current_graph }

        changed_records = new_graph_set - old_graph_set

        for record in changed_records:
            self.controller_to_inferencer.put( record )
            self.controller_to_model_manager.put( record )

        await self.set_current_graph( new_graph )

    async def load_current_filesystem_graph(self):
        from aiofile import async_open
        async with async_open(self.config.filesystem_state_file, 'r') as afp:
            records_text = await afp.read()

        records_json = json.loads( records_text )

        records = []
        for record_dict in records_json:
            record = FileRecord.from_dict( record_dict )
            records.append( record )

        self.current_graph = records

    async def save_current_filesystem_graph(self):
        from aiofile import async_open
        async with async_open(self.config.filesystem_state_file, 'w') as afp:
            records_dict = [ record.to_dict() for record in self.current_graph ]

            await afp.write( json.dumps( records_dict ) )

    async def set_current_graph(self, current_graph: List[ "FileRecord" ]):
        self.current_graph = current_graph
        await self.save_current_filesystem_graph()


