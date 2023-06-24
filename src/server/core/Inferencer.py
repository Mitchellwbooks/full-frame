import asyncio
from multiprocessing import Process, Queue
from typing import List

import numpy as np
import onnxruntime
from torch import onnx
from torchvision.transforms import transforms

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
    model_labels: List[str]

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

    def run(self):
        asyncio.run(self.async_run())

    async def async_run(self):
        self.inferencer_to_model_manager.put({
            'type': 'model_request'
        })
        while self.model_manager_to_inferencer.empty():
            # Wait for a response back before continuing
            await asyncio.sleep(5)

        await self.load_model_manager_messages()

        # Begin execution loop
        while True:
            processed_controller_message = await self.load_controller_message()
            processed_manager_message = await self.load_model_manager_messages()

            if processed_controller_message is False and processed_manager_message is False:
                await asyncio.sleep(10)

    async def load_model_manager_messages(self):
        processed_message = False
        while self.model_manager_to_inferencer.qsize() > 0:
            processed_message = True
            message = self.model_manager_to_inferencer.get()

            if message['type'] == 'onnx_model':
                self.onnx_model = message['onnx_model']
                self.model_labels = message['model_labels']
                onnx.checker.check_model(self.onnx_model)
                self.model_runtime = onnxruntime.InferenceSession(self.onnx_model)
        return processed_message

    async def load_controller_message(self):
        """
        Loads a message from the controller message queue.
        Performs inference
        Distributes Results

        Returns:
            bool: whether a message was received and processed.
        """

        if self.controller_to_inferencer.empty():
            return False

        file_record: FileRecord = self.controller_to_inferencer.get()

        # Process image into model-compatible format.
        image = file_record.load_pil_image()
        image = image.convert('RGB')
        preprocessing = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        processed_image = preprocessing(image)

        # Inference
        results = self.model_runtime.run(
            None,
            input_feed={
                'data': [
                    processed_image.numpy()
                ]
            }
        )

        # Process Results
        output = results[0].flatten()
        output = self.softmax(output)
        sorted_predictions = np.argsort(-output)

        # Organize Predictions
        predictions = []
        for match_index in sorted_predictions:
            label = self.model_labels[match_index]
            confidence = output[match_index]
            predictions.append({
                'label': label,
                'confidence': confidence
            })

        file_record.add_label_inferences(
            predictions,
            self.model_runtime.get_modelmeta()
        )
        self.inferencer_to_model_manager.put({
            'file_record': file_record,
            'inferences': predictions,
            'model_metadata': self.model_runtime.get_modelmeta()
        })

        return True

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
