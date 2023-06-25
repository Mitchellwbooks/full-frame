import asyncio
from multiprocessing import Process

import pandas as pd

from core.library.FileRecord import FileRecord


class ModelManager( Process ):
    """
    Abstracts over model versioning
    """

    def __init__(
            self,
            controller_to_model_manager,
            model_manager_to_controller,
            inferencer_to_model_manager,
            model_manager_to_inferencer
    ):
        super().__init__()
        self.controller_to_model_manager = controller_to_model_manager
        self.model_manager_to_controller = model_manager_to_controller
        self.inferencer_to_model_manager = inferencer_to_model_manager
        self.model_manager_to_inferencer = model_manager_to_inferencer
        self.file_frame = pd.DataFrame()
        self.file_frame.set_index('raw_file_path')

    def run( self ):
        pass

    def async_run(self):
        while True:
            await self.load_inferencer_messages()
            await self.load_controller_messages()
            await self.train()

    async def load_inferencer_messages(self):
        if self.inferencer_to_model_manager.empty():
            return False

        queue_size = self.inferencer_to_model_manager.qsize()
        file_records_put_coros = []
        for message in range( queue_size ):
            file_record: FileRecord = self.inferencer_to_model_manager.get()
            file_records_put_coros.append( self.put_file_record(file_record) )

        # :todo: don't load here
        await asyncio.gather( * file_records_put_coros )

    async def load_controller_messages(self):
        if self.controller_to_model_manager.empty():
            return False

        queue_size = self.controller_to_model_manager.qsize()
        file_records_put_coros = []
        for message in range(queue_size):
            file_record: FileRecord = self.controller_to_model_manager.get()
            file_records_put_coros.append(self.put_file_record(file_record))

        # :TODO: don't load here
        await asyncio.gather(*file_records_put_coros)

    async def put_file_record(self, file_record):
        # :TODO: Handle existing records
        self.file_frame.append( await file_record.to_dict() )

    async def train(self):
        """
        https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
        :return:
        """
        pass