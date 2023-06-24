import asyncio
from queue import Queue
from unittest import TestCase

import onnxruntime

from core.library.FileRecord import FileRecord


class TestInferencer( TestCase ):

    def test_load_controller_message(self):
        from core.Inferencer import Inferencer

        """ Case 1: Image of cheetah """
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

        file_record = asyncio.run( FileRecord.init( 'testdata/cheetah.png', '.png' ) )
        controller_to_inferencer.put( file_record )

        inferencer.model_runtime = onnxruntime.InferenceSession(
            '../../onnx_models/resnet50-v2-7.onnx',
            providers=[
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        )

        with open('../../onnx_models/resnet-50-v2-7.labels') as csvfile:
            labels = csvfile.read()
            labels = labels.split('\n')

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
        self.assertTrue( 'cheetah' in first_result['label'], 'We expected the top label to be a cheetah' )
        self.assertTrue( first_result['confidence'] > .8, 'We expected the confidence to be higher that 80%' )
