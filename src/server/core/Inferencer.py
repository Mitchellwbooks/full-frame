import asyncio
from multiprocessing import Process, Queue

import numpy as np
import onnxruntime
import pandas as pd
from torchvision.transforms import transforms

from core.library.Config import Config
from core.library.FileRecord import FileRecord


class Inferencer(Process):
    """
    Abstracts over model execution
    """
    inferencer_to_model_manager: Queue
    model_manager_to_inferencer: Queue
    controller_to_inferencer: Queue
    inferencer_to_controller: Queue

    onnx_model: str
    model_runtime: "onnxruntime.InferenceSession"
    model_labels: pd.DataFrame

    def __init__(
            self,
            controller_to_inferencer,
            inferencer_to_controller,
            inferencer_to_model_manager,
            model_manager_to_inferencer
    ):
        super().__init__()
        self.controller_to_inferencer = controller_to_inferencer
        self.inferencer_to_controller = inferencer_to_controller
        self.inferencer_to_model_manager = inferencer_to_model_manager
        self.model_manager_to_inferencer = model_manager_to_inferencer
        self.model_runtime = None
        self.file_dict = {}
        self.file_ids_to_inference = []
        self.config = Config()
        self.continue_processing = True

    def run(self):
        asyncio.run(self.async_run())

    async def async_run(self):
        await asyncio.gather(
            self.run_inferences(),
            self.run_read_controller_messages(),
            self.run_read_model_manager_messages()
        )

    async def run_read_controller_messages(self):
        while self.continue_processing:
            if self.controller_to_inferencer.qsize() == 0:
                await asyncio.sleep(10)
                continue

            message = self.controller_to_inferencer.get()

            if message['topic'] == 'discovered_file':
                # Refresh local file to initialize with changes.
                file_record = message['file_record']
                file_lookup = hash( file_record )
                local_record = self.file_dict[ file_lookup ]
                self.file_dict[ file_lookup ] = await FileRecord.init( local_record.raw_file_path )
                if file_lookup not in self.file_ids_to_inference:
                    self.file_ids_to_inference.append(file_lookup)

            if message['topic'] == 'removed_file':
                # Refresh local file to initialize with changes.
                file_record = message['file_record']
                file_lookup = hash( file_record )
                del self.file_dict[ file_lookup ]
                if file_lookup in self.file_ids_to_inference:
                    self.file_ids_to_inference.remove(file_lookup)

            if message['topic'] == 'metadata_file_changed':
                # Refresh local file to initialize with changes.
                file_record = message['file_record']
                file_lookup = hash( file_record )
                if file_lookup not in self.file_ids_to_inference:
                    self.file_ids_to_inference.append(file_lookup)

    async def run_read_model_manager_messages(self):
        while self.continue_processing:
            if self.model_manager_to_inferencer.qsize() == 0:
                await asyncio.sleep(10)
                continue

            message = self.model_manager_to_inferencer.get()

            if message['topic'] == 'onnx_model_created':
                # self.onnx_runtime = message['onnx_model']
                self.model_labels = message['model_labels']
                self.model_runtime = onnxruntime.InferenceSession(
                    message['onnx_model_path'],
                    providers=[
                        'TensorrtExecutionProvider',
                        'CUDAExecutionProvider',
                        'CPUExecutionProvider'
                    ]
                )

                # Run inferences on all our tracked files
                for file_lookup in self.file_dict.keys():
                    if file_lookup not in self.file_ids_to_inference:
                        self.file_ids_to_inference.append(file_lookup)

    async def run_inferences(self):
        while self.model_runtime is None:
            print('Inferencer: no model available yet')
            await asyncio.sleep(30)
            continue

        while self.continue_processing:
            if len( self.file_ids_to_inference ) == 0:
                await asyncio.sleep(10)
                continue

            file_lookup = self.file_ids_to_inference.pop()
            file_record = self.file_dict[ file_lookup ]
            await self.inference_file( file_record )

    async def inference_file(self, file_record: FileRecord ):
        """
        Loads a message from the controller message queue.
        Performs inference
        Distributes Results

        Returns:
            bool: whether a message was received and processed.
        """
        print( f'Inferencer: Processing {file_record.raw_file_path}')

        # Process image into model-compatible format.
        # :TODO: Cache feature map https://github.com/Mitchellwbooks/full-frame/issues/14
        image = await file_record.load_pil_image()
        preprocessing = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        processed_image = preprocessing(image)

        # Inference
        results = self.model_runtime.run(
            None,
            input_feed={
                'input': [
                    processed_image.numpy()
                ]
            }
        )

        # Process Results
        output = results[0].flatten()
        # :TODO: I think we may want a sigmoid output function here instead.
        # See how model manager scores accuracy
        output = self.softmax(output)
        sorted_predictions = np.argsort(-output)

        # Organize Predictions
        labels = []
        predictions = []
        for match_index in sorted_predictions:
            label = self.model_labels.iloc[match_index]
            confidence = output[match_index]
            predictions.append({
                'label': label['labels'],
                'confidence': confidence
            })

            if confidence > self.config.confidence_threshold:
                print( label['labels'] )
                labels.append( label['labels'] )

        await file_record.add_label_inferences(
            labels
        )

        self.inferencer_to_model_manager.put({
            'topic': 'inference_made',
            'record': file_record
        })

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
