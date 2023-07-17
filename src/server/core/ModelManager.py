import asyncio
import os
from multiprocessing import Process

import onnx
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelAUROC
from torchvision.transforms import transforms, RandomRotation

from core.library.Constants import INCORRECT_INFERENCES_SUBJECT
from core.library.FileRecord import FileRecord


class ModelManager(Process):
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

        self.file_dict = {}
        self.file_ids_pending_training = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.base_model_path = f'{dir_path}/../onnx_models/resnet_50.onnx'
        self.updated_model_path = f'{dir_path}/../onnx_models/resnet_50_updated.onnx'
        self.model_label_path = f'{dir_path}/../onnx_models/resnet_50_updated_labels.csv'
        self.continue_processing = True

    def run(self):
        asyncio.run(self.async_run())

    async def async_run(self):
        await asyncio.gather(
            self.run_read_controller_messages(),
            self.run_model_training()
        )

    async def run_read_controller_messages(self):
        while self.continue_processing:
            if self.controller_to_model_manager.qsize() == 0:
                await asyncio.sleep(10)
                continue

            message = self.controller_to_model_manager.get()

            print( f'ModelManager: Processing Message {message}' )

            if message['topic'] == 'discovered_file':
                # Refresh local file to initialize with changes.
                file_record = message['file_record']
                file_lookup = hash( file_record )
                self.file_dict[ file_lookup ] = await FileRecord.init( file_record.raw_file_path )
                if file_lookup not in self.file_ids_pending_training:
                    self.file_ids_pending_training.append(file_lookup)

            if message['topic'] == 'removed_file':
                # Refresh local file to initialize with changes.
                file_record = message['file_record']
                file_lookup = hash( file_record )
                del self.file_dict[ file_lookup ]
                if file_lookup in self.file_ids_pending_training:
                    self.file_ids_pending_training.remove(file_lookup)

            if message['topic'] == 'metadata_file_changed':
                # Refresh local file to initialize with changes.
                file_record = message['file_record']
                file_lookup = hash( file_record )
                if file_lookup not in self.file_ids_pending_training:
                    self.file_ids_pending_training.append(file_lookup)

    async def run_model_training(self):
        while self.continue_processing:
            print( f'ModelManager: Files Pending Training {self.file_ids_pending_training}' )
            if len( self.file_ids_pending_training ) > 10:
                self.file_ids_pending_training = []
                updated_model = await self.train( onnx.load( self.base_model_path ) )
                if updated_model is not None:
                    await self.send_new_model()

    async def send_new_model( self ):
        # new_model = onnx.load( self.updated_model_path )
        # new_model = onnxruntime.InferenceSession(
        #     self.updated_model_path,
        #     providers=[
        #         'TensorrtExecutionProvider',
        #         'CUDAExecutionProvider',
        #         'CPUExecutionProvider'
        #     ]
        # )
        model_labels = pd.read_csv( self.model_label_path )
        self.model_manager_to_inferencer.put( {
            'type': 'onnx_model',
            # 'onnx_model': new_model,
            'onnx_model_path': self.updated_model_path,
            'model_labels': model_labels
        } )

    async def train(self, input_model):
        """
        https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
        https://github.com/ENOT-AutoDL/onnx2torch

        https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        :return:
        """
        print( 'Starting Training')
        from onnx2torch import convert

        file_metadata_frame, labels_frame = await self.load_file_metadata_frame()

        if len( labels_frame ) == 0:
            # No labels to train
            print( 'no labels to train')
            return None

        # We will confuse training if we have a lot of images with no labels at all. So only train on files with labels.
        files_with_labels = file_metadata_frame[ file_metadata_frame['has_labels'] == True ]

        df_train = files_with_labels.sample(frac=0.8, random_state=1)
        df_test = files_with_labels.drop(df_train.index)

        torch_model = convert(input_model)

        for param in torch_model.parameters():
            param.requires_grad = False

        # Adapting the structure: We don't squish the output to the top label, but rather the set of top labels.
        # fc/Gemm is the original fully connected output of the model, so we are adapting the output to our needs
        # :TODO: Cache feature map https://github.com/Mitchellwbooks/full-frame/issues/14
        existing_fully_connected = torch_model.__getattr__('fc/Gemm')
        for param in existing_fully_connected.parameters():
            param.requires_grad = True

        torch_model.__setattr__(
            'fc/Gemm',
            nn.Sequential(
                existing_fully_connected,
                # We are squishing the outputs, so we have a close binary hot encoding of labels.
                nn.Sigmoid(),
                nn.Linear(
                    # out_features is 1000; This could be limiting since it is only 1000 features, but the rest are 2000
                    in_features=existing_fully_connected.out_features,
                    out_features=len(labels_frame)
                )
            )
        )

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(torch_model.parameters(), lr=learning_rate)

        # Again: We don't squish the output to the top label, but rather the set of top labels.
        criterion = nn.BCEWithLogitsLoss(
            # We are adjusting the loss function to favor false positives.
            # Theory: A user can remove inferences easier than the program not suggesting them at all.
            #         See the inverse effect this will have here:
            #           https://github.com/Mitchellwbooks/full-frame/issues/11
            # ERROR: this if for weighting the number of examples of a label not this ^
            # pos_weight=torch.tensor([1.2] * len(labels_frame))
        )

        updated_model = await self.train_model(
            torch_model,
            criterion,
            optimizer,
            labels_frame,
            df_train,
            df_test
        )

        example_image = file_metadata_frame['tensor_image'].iloc[0]

        torch.onnx.export(
            updated_model,  # model being run
            example_image.unsqueeze(0),  # model input (or a tuple for multiple inputs)
            self.updated_model_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=13,  # the ONNX version to export the model to
            do_constant_folding=False,  # whether to execute constant folding for optimization
            input_names=['input'],  # the model's input names
            output_names=['output'],  # the model's output names
            dynamic_axes={
                'input': {
                    0: 'batch_size'
                },  # variable length axes
                'output': {
                    0: 'batch_size'
                }
            }
        )

        labels_frame.to_csv( self.model_label_path )
        print( 'Produced new model')

        return updated_model

    async def train_model(self, model, criterion, optimizer, all_labels, training, validation, num_epochs=8):
        """
        Read More:
        https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch
        :return:
        """
        randomization = transforms.Compose([
            # Randomize the rotation by 20 degrees to combat memorization.
            RandomRotation((-20, 20))
        ])

        dataloaders = {
            'train': torch.utils.data.DataLoader(
                CustomImageDataset(training, all_labels, randomization),
                batch_size=10,
                shuffle=True,
                num_workers=4
            ),
            'validation': torch.utils.data.DataLoader(
                CustomImageDataset(validation, all_labels, randomization),
                batch_size=10,
                shuffle=True,
                num_workers=4
            )
        }

        # :TODO: Stop learning due to learning deceleration
        #        https://github.com/Mitchellwbooks/full-frame/issues/12
        for epoch in range(num_epochs):
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                accuracy = Accuracy( task="multilabel", num_labels=len(all_labels), average=None)
                auroc_score = MultilabelAUROC( num_labels=len(all_labels), validate_args=False)

                for inputs, labels in dataloaders[phase]:
                    outputs = model(inputs)
                    # :TODO: Add negative label examples
                    #        https://github.com/Mitchellwbooks/full-frame/issues/11
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    outputs = nn.Sigmoid()(outputs)
                    outputs = (outputs > torch.tensor([0.5])).float() * 1

                    batch_accuracy = accuracy( outputs, labels )
                    batch_auroc = auroc_score( outputs, labels )
                    # print(
                    #     f'      loss: {running_loss}\n'
                    #     f'      acc: {batch_accuracy}\n'
                    #     f'      auroc: {batch_auroc}\n'
                    # )

                epoch_loss = running_loss / len(dataloaders[phase])
                output = \
                    f'{phase} \n' \
                    f'    loss: {epoch_loss}\n' \
                    f'    acc: {accuracy.compute()}\n' \
                    f'    auroc: {auroc_score.compute()}\n'
                print(output)
        return model

    async def load_file_metadata_frame(self):
        from core.library.Constants import USER_CREATED_SUBJECT, CONFIRMED_INFERENCES_SUBJECT
        all_labels = []
        file_records = []
        file_ids = self.file_dict.keys()
        for file_id in file_ids:
            file_record = self.file_dict[ file_id ]
            pil_image = await file_record.load_pil_image()

            labels = await file_record.load_xmp_subject( USER_CREATED_SUBJECT )
            labels += await file_record.load_xmp_subject( CONFIRMED_INFERENCES_SUBJECT )
            negative_labels = await file_record.load_xmp_subject( INCORRECT_INFERENCES_SUBJECT )

            all_labels += labels
            preprocessing = transforms.Compose([
                transforms.ToTensor(),
                # Resnet 50 only supports 224 square input images.
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    # oooo magic numbers ... pulled from every other example of turning a pil image into a tensor.
                    # Tweaking colors to fit between 0-1, models perform better when pixels to be restrained to a limited range.
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
            image = preprocessing(pil_image)

            file_records.append({
                'file_path': file_record.raw_file_path,
                'file_record': file_record,
                # :TODO: Prevent memory issue during training for large sets of images
                #        https://github.com/Mitchellwbooks/full-frame/issues/13
                'tensor_image': image,
                'labels': labels,
                'negative_labels': negative_labels,
                'has_labels': len( labels ) > 0 or len( negative_labels ) > 0
            })

        dataframe = pd.DataFrame(file_records)

        dataframe['thumbnail_path'] = dataframe['file_record'].apply(lambda x: x.thumbnail_path)

        labels_frame = pd.DataFrame({'labels': all_labels})
        unique_labels = labels_frame['labels'].unique()
        labels_frame = pd.DataFrame({'labels': unique_labels})

        def labels_truth( row ):
            labels_truth_list = []
            for label in labels_frame['labels']:
                if label in row[ 'labels' ]:
                    labels_truth_list.append( 1 )
                elif label in row[ 'negative_labels' ]:
                    labels_truth_list.append( 0 )
                else:
                    labels_truth_list.append( 0 )

            return torch.tensor(labels_truth_list, dtype=torch.float32)

        dataframe['label_tensor'] = dataframe.apply( labels_truth, axis = 1  )

        return dataframe, labels_frame


class CustomImageDataset(Dataset):
    """
    Readmore
    https://stackoverflow.com/questions/66446881/how-to-use-a-pytorch-dataloader-for-a-dataset-with-multiple-labels
    """

    def __init__(self, files_dataframe, labels, transform=None):
        self.files_dataframe = files_dataframe
        self.transform = transform
        self.labels = labels

    def __len__(self):
        """
        Return number of rows in dataset
        :return:
        """
        return len(self.files_dataframe)

    def __getitem__(self, idx: int):
        """
        Loads image from dataset
        :param idx:
        :return:
        """
        file_row = self.files_dataframe.iloc[idx]
        image = file_row['tensor_image']
        if self.transform:
            image = self.transform(image)
        return image, file_row['label_tensor']
