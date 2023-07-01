import asyncio
from multiprocessing import Process

import onnx
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelAUROC
from torchvision.transforms import transforms, RandomRotation

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

        self.file_list = []
        self.base_model_path = '../onnx_models/resnet_50.onnx'
        self.updated_model_path = '../onnx_models/resnet_50_updated.onnx'
        self.model_label_path = '../onnx_models/resnet_50_updated_labels.csv'

    def run(self):
        asyncio.run(self.async_run())

    async def async_run(self):
        while True:
            # :TODO: being super lazy here, should wait for a certain amount of data before training.
            await asyncio.sleep( 60 )

            await self.load_inferencer_messages()
            await self.load_controller_messages()

            await self.train( onnx.load( self.base_model_path ) )
            await self.send_new_model()
            # :TODO: being super lazy here, should wait for a certain amount of data before training.
            await asyncio.sleep( 60 )

    async def load_inferencer_messages(self):
        if self.inferencer_to_model_manager.empty():
            return False

        queue_size = self.inferencer_to_model_manager.qsize()
        for message in range(queue_size):
            file_record: FileRecord = self.inferencer_to_model_manager.get()
            self.file_list.append(file_record.raw_file_path)

    async def load_controller_messages(self):
        if self.controller_to_model_manager.empty():
            return False

        queue_size = self.controller_to_model_manager.qsize()
        for message in range(queue_size):
            file_record: FileRecord = self.controller_to_model_manager.get()
            self.file_list.append(file_record.raw_file_path)

    async def send_new_model( self ):
        new_model = onnx.load( self.updated_model_path )
        model_labels = pd.read_csv( self.model_label_path )
        self.model_manager_to_inferencer.send( {
            'onnx_model': new_model,
            'model_labels': model_labels
        } )

    async def train(self, input_model):
        """
        https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
        https://github.com/ENOT-AutoDL/onnx2torch

        https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        :return:
        """
        from onnx2torch import convert

        file_metadata_frame, labels_frame = await self.load_file_metadata_frame()

        df_train = file_metadata_frame.sample(frac=0.8, random_state=1)
        df_test = file_metadata_frame.drop(df_train.index)

        torch_model = convert(input_model)

        for param in torch_model.parameters():
            param.requires_grad = False

        # Adapting the structure: We don't squish the output to the top label, but rather the set of top labels.
        # fc/Gemm is the original fully connected output of the model, so we are adapting the output to our needs
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
            pos_weight=torch.tensor([1.2] * len(labels_frame))
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

        return updated_model

    async def train_model(self, model, criterion, optimizer, all_labels, training, validation, num_epochs=2):
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
                    print( outputs )
                    print( labels )
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

                epoch_loss = running_loss / len(dataloaders[phase])
                output = \
                    f'{phase} \n' \
                    f'    loss: {epoch_loss}\n' \
                    f'    acc: {accuracy.compute()}\n' \
                    f'    auroc: {auroc_score.compute()}\n'
                print(output)
        return model

    async def load_file_metadata_frame(self):
        all_labels = []
        file_records = []
        for file_path in self.file_list:
            file_record = await FileRecord.init(
                file_path
            )
            pil_image = await file_record.load_pil_image()
            labels = await file_record.load_xmp_subject()
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
                'file_path': file_path,
                'file_record': file_record,
                # :TODO: We are loading all the images into memory. This could cause severe memory problems.
                'tensor_image': image,
                'labels': labels
            })

        dataframe = pd.DataFrame(file_records)

        dataframe['thumbnail_path'] = dataframe['file_record'].apply(lambda x: x.thumbnail_path)

        labels_frame = pd.DataFrame({'labels': all_labels})
        unique_labels = labels_frame['labels'].unique()
        labels_frame = pd.DataFrame({'labels': unique_labels})

        def labels_truth(labels):
            labels_truth_list = []
            for label in labels_frame['labels']:
                if label in labels:
                    labels_truth_list.append(1)
                else:
                    labels_truth_list.append(0)

            return torch.tensor(labels_truth_list, dtype=torch.float32)

        dataframe['label_tensor'] = dataframe['labels'].apply( labels_truth )

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
