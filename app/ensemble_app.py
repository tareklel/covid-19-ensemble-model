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


class Clahe:
    def __init__(self):
        self.image_list = None
        self.image_path = None

    def get_image_list(self, path):
        """
        get path of images and return list of image paths
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

    def rm_clahe(self):
        """del clahe folder"""
        os.remove('clahe_dir/clahe_dir')
        os.renive('clahe_dir')




class ModelResults:
    """"""
    def __init__(self):
        self.model_path = 'models'
        self.model_dict = {'vgg': 'models/vgg.pth',
                           'densenet': 'models/densenet.pth',
                           'resnet': 'models/resnext.pth'
                          }

    def prepare_dataset(self):
        """Apply transformations for dataset and get filenames"""
        dataset = ImageFolder('clahe_dir',
                                       transform=transforms.Compose([
                                           transforms.Resize(512),
                                           transforms.CenterCrop(480),
                                           transforms.ToTensor(),
                                       ]))
        test_dl = DataLoader(dataset, batch_size=20)
        self.file_names = [x[0].split('/')[-1] for x in test_dl.dataset.samples]
        self.test_dl = DeviceDataLoader(test_dl, self.get_default_device())


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

    def predict(self, model):
        """Return prediction outputs """
        model.eval()
        outputs = [model.predict(batch) for batch in self.test_dl]
        print(f'{model.network.__class__.__name__} outputs ready')
        return outputs

    def output_to_table(self, model, results):
        """takes results and model, creates DataFrame"""
        name = model.network.__class__.__name__
        prediction_list = []
        for row in results:
            for num in row.tolist():
                prediction_list.append(num)

        df = pd.DataFrame(prediction_list, columns=[f'{name}_label_0', f'{name}_label_1', f'{name}_label_2'])
        return df

    def get_ensemble_table(self):
        df_list = []
        for m in self.model_dict:
            model = self.load_model(m)
            out = self.predict(model)
            df = self.output_to_table(model, out)
            df_list.append(df)

        df_concat = pd.concat(df_list, axis=1)
        df_concat['filename'] = self.file_names
        return df_concat




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


if __name__ =="__main__":
    clahe = Clahe()
    clahe.get_image_list(DEMO_DIR)
    clahe.convert_clahe()
    # master.prepare_dataset()
    #print(master.get_ensemble_table())














