import asyncio
from multiprocessing import Process

import onnx
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torchvision.transforms import transforms

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

        self.file_list = []

        self.current_model_onnx = onnx.load( 'onnx_models/resnet50-v2.7.onnx' )

    def run( self ):
        asyncio.run( self.async_run() )

    async def async_run(self):
        while True:
            await self.load_inferencer_messages()
            await self.load_controller_messages()
            await self.train( self.current_model_onnx )

    async def load_inferencer_messages(self):
        if self.inferencer_to_model_manager.empty():
            return False

        queue_size = self.inferencer_to_model_manager.qsize()
        for message in range( queue_size ):
            file_record: FileRecord = self.inferencer_to_model_manager.get()
            self.file_list.append( file_record.raw_file_path )

    async def load_controller_messages(self):
        if self.controller_to_model_manager.empty():
            return False

        queue_size = self.controller_to_model_manager.qsize()
        for message in range(queue_size):
            file_record: FileRecord = self.controller_to_model_manager.get()
            self.file_list.append( file_record.raw_file_path )

    async def train(self, input_model):
        """
        https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
        https://github.com/ENOT-AutoDL/onnx2torch

        https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        :return:
        """
        from onnx2torch import convert

        file_metadata_frame = await self.load_file_metadata_frame()

        df_train = file_metadata_frame.sample( frac = 0.8, random_state = 1 )
        df_test = file_metadata_frame.drop( df_train.index )

        torch_model = convert( self.current_model_onnx )

        for param in torch_model.parameters():
            param.requires_grad = False

        # Adapting the structure: We don't squish the output to the top label, but rather the set of top labels.
        torch_model.fc = nn.Sequential(
            nn.Dropout( p=0.2 ),
            nn.Linear(
                in_features=torch_model.fc.in_features,
                out_features=file_metadata_frame['labels'].nunique()
            )
        )
        torch_model.sigm = nn.Sigmoid()
        torch_model.forward = lambda s, x: s.sigm(s.base_model(x))

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(torch_model.parameters(), lr=learning_rate)

        # Again: We don't squish the output to the top label, but rather the set of top labels.
        criterion = nn.BCELoss()

        updated_model = await self.train_model( torch_model, criterion, optimizer, df_train, df_test )

        return updated_model

    @staticmethod
    async def train_model(model, criterion, optimizer, training, validation, num_epochs=3):
        """
        Read More:
        https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch
        :return:
        """
        dataloaders = {
            'train': training,
            'validation': validation
        }
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects.double() / len(dataloaders[phase])

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
        return model

    async def load_file_metadata_frame( self ):

        file_records = []
        for file_path in self.file_list:
            file_records.append( {
                'file_path': file_path,
                'file_record': await FileRecord.init(
                    file_path
                )
            } )

        dataframe = pd.DataFrame( file_records )

        dataframe[ 'thumbnail_path' ] = dataframe['file_record'].apply( lambda x: x.thumbnail_path )
        dataframe[ 'labels' ] = dataframe['file_record'].apply( lambda x: x.thumbnail_path )
        # dataframe[ 'label_ids' ] = :TODO:

        return dataframe


class CustomImageDataset(Dataset):
    """
    Readmore
    https://stackoverflow.com/questions/66446881/how-to-use-a-pytorch-dataloader-for-a-dataset-with-multiple-labels
    """
    def __init__(self, files_dataframe, transform=None):
        self.files_dataframe = files_dataframe
        self.transform = transform

    def __len__(self):
        """
        Return number of rows in dataset
        :return:
        """
        return len( self.files_dataframe )

    def __getitem__(self, idx: int):
        """
        Loads image from dataset
        :param idx:
        :return:
        """
        file_row = self.files_dataframe.iloc[idx]
        image = read_image( file_row['thumbnail_path'] )
        preprocessing = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image = preprocessing( image )
        labels = file_row['label_ids']
        if self.transform:
            image = self.transform(image)
        return image, labels
