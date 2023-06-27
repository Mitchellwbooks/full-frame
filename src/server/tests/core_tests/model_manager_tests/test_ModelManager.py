import asyncio
import os
from queue import Queue
from unittest import TestCase

import onnx
import onnxruntime

from core.library.FileRecord import FileRecord


class TestInferencer( TestCase ):

    def test_load_controller_message(self):
        from core.ModelManager import ModelManager

        """ Case 1: Image of cheetah """
        inferencer_to_model_manager = Queue()
        model_manager_to_inferencer = Queue()
        controller_to_model_manager = Queue()
        model_manager_to_controller = Queue()

        model_manager = ModelManager(
            controller_to_model_manager,
            model_manager_to_controller,
            inferencer_to_model_manager,
            model_manager_to_inferencer,
        )

        input_model = onnx.load( 'onnx_models/resnet50-v2.7.onnx' )

        antelope_files = []
        for file in os.listdir('testdata/pictures/antelope' ):
            if file.endswith('.jpg'):
                antelope_files.append( file )

        model_manager.file_list = antelope_files

        FileRecord.config.thumbnail_path = 'testdata/thumbnails'
        if os.path.exists( FileRecord.config.thumbnail_path ) is False:
            os.mkdir( FileRecord.config.thumbnail_path )

        ''' Base Assertions '''

        ''' Function Call '''
        output_model = asyncio.run( model_manager.train( input_model ) )

        ''' Assertions '''

