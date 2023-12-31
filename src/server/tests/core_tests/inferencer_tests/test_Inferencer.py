import asyncio
import os
from queue import Queue
from unittest import TestCase

import onnxruntime
import pandas as pd

from core.library.FileRecord import FileRecord


class TestInferencer( TestCase ):

    def test_load_controller_message(self):
        from core.Inferencer import Inferencer

        """ Case 1: Image of antelope """
        inferencer_to_model_manager = Queue()
        model_manager_to_inferencer = Queue()
        controller_to_inferencer = Queue()
        inferencer_to_controller = Queue()

        inferencer = Inferencer(
            controller_to_inferencer,
            inferencer_to_controller,
            inferencer_to_model_manager,
            model_manager_to_inferencer,
        )

        FileRecord.config.thumbnail_path = '../testdata/thumbnails'
        if os.path.exists( FileRecord.config.thumbnail_path ) is False:
            os.mkdir( FileRecord.config.thumbnail_path )

        file_record = asyncio.run(FileRecord.init('../testdata/pictures/antelope/0a37838e99.jpg'))
        controller_to_inferencer.put( file_record )

        inferencer.model_runtime = onnxruntime.InferenceSession(
            '../../../onnx_models/resnet_50_updated.onnx',
            providers=[
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        )

        labels = pd.read_csv( '../../../onnx_models/resnet_50_updated_labels.csv' )
        inferencer.model_labels = labels

        ''' Function Call '''
        message_processed = asyncio.run( inferencer.load_controller_message() )

        ''' Assertions '''
        self.assertTrue( message_processed, 'We expected a message to be processed.' )
        self.assertTrue( controller_to_inferencer.empty(), 'We expected the message to be consumed.' )

        self.assertTrue( inferencer_to_model_manager.qsize() == 1, 'We expected a result message to be produced.')

        # Assert Top Result is a cheetah
        result = inferencer_to_model_manager.get()

        first_result = result['inferences'][ 0 ]
        self.assertTrue( 'antelope' in first_result['label'], 'We expected the top label to be a cheetah' )
        self.assertTrue( first_result['confidence'] > .8, 'We expected the confidence to be higher that 80%' )
