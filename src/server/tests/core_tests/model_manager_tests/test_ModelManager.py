import asyncio
import os
from queue import Queue
from unittest import TestCase

import onnx
from onnx import version_converter

from core.library.FileRecord import FileRecord


class TestModelManager( TestCase ):

    def test_train(self):
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

        model_manager.updated_model_path = '../../../onnx_models/resnet_50_updated.onnx'
        model_manager.model_label_path = '../../../onnx_models/resnet_50_updated_labels.csv'
        input_model = onnx.load( '../../../onnx_models/resnet_50.onnx' )
        input_model = version_converter.convert_version( input_model, 13 )

        file_list = []
        for file in os.listdir('../testdata/pictures/antelope'):
            if file.endswith('.jpg'):
                file_list.append( f'../testdata/pictures/antelope/{file}' )

        for file in os.listdir('../testdata/pictures/bison'):
            if file.endswith('.jpg'):
                file_list.append( f'../testdata/pictures/bison/{file}' )

        model_manager.file_list = file_list

        FileRecord.config.thumbnail_path = '../testdata/thumbnails'
        if os.path.exists( FileRecord.config.thumbnail_path ) is False:
            os.mkdir( FileRecord.config.thumbnail_path )

        ''' Base Assertions '''

        ''' Function Call '''
        output_model = asyncio.run( model_manager.train( input_model ) )

        ''' Assertions '''

