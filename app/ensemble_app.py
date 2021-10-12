import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
import PIL
import ast
import sklearn
import sklearn.model_selection
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
import torchvision.models as models
from torchvision.datasets import ImageFolder
import albumentations as a
from model_net import ModelBase, ModelNet


DEMO_DIR = 'test_images/normal'


class EnsembleModel:
    def __init__(self):
        self.model_path = 'models'
        self.clahe_dir = 'clahe_dir'
        self.model_dict = {'vgg': 'models/vgg.pth',
                           'densenet': 'models/densenet.pth',
                           'resnet': 'models/resnext.pth'
                          }

    def get_image_list(self, path):
        """
        get path of images and return list of image path
        """
        self.image_path = path
        self.image_list = []
        for impth in os.listdir(self.image_path):
            path = self.image_path + '/' + impth
            if os.path.isfile(self.image_path + '/' + impth):
                self.image_list.append(path)

    def convert_clahe(self):
        """
        create clahe folder,  process images with clahe to folder
        """
        os.mkdir('clahe_dir')
        os.mkdir('clahe_dir/clahe_dir')
        for file in self.image_list:
            image_name = file.split('/')[-1]
            img = cv2.imread(file, 0)
            clahe = cv2.createCLAHE(tileGridSize=(8,8))
            cl1 = clahe.apply(img)
            cv2.imwrite('clahe_dir/clahe_dir/' + image_name, cl1)

    def get_default_device(self):
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def load_model(self, name):
        """loads deep learning model"""
        filepath = self.model_dict[name]
        checkpoint = torch.load(filepath, self.get_default_device())
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        print(f'{model.network.__class__.__name__} loaded')
        return model

    @torch.no_grad()
    def test_predict(self, model, test_loader):
        model.eval()
        # perform testing for each batch
        outputs = [model.validation_step(batch) for batch in test_loader]
        results = model.test_prediction(outputs)
        print('test_loss: {:.4f}, test_acc: {:.4f}'
              .format(results['test_loss'], results['test_acc']))

        return results['test_preds'], results['test_labels'], results['out']


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            # yield will stop here, perform other steps,
            # and the resumes to the next loop/batch
            yield to_device(b,
                            self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def create_test_dataset(data_dir):
    test_dataset = ImageFolder(data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(512),
                                   transforms.CenterCrop(480),
                                   transforms.ToTensor(),
                               ]))
    return test_dataset


if __name__ =="__main__":
    master = EnsembleModel()
    #master.get_image_list(DEMO_DIR)
    #master.convert_clahe()
    test_dataset = create_test_dataset('clahe_dir')
    test_dl = DataLoader(test_dataset, batch_size=(20))
    test_dl = DeviceDataLoader(test_dl, master.get_default_device())
    model = master.load_model('densenet')
    model(test_dl)














