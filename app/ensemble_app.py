import pandas as pd
import os
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model_net import ModelBase, ModelNet
from sklearn.linear_model import LogisticRegression
import sys
import pickle


DEMO_DIR = 'test_images/pneumonia'


class Clahe:
    """converts and removes CLAHE converted images used in predictions"""
    def __init__(self):
        self.image_list = None
        self.image_path = None
        self.image_list = []

    def get_image_list(self, path):
        """get path of images and return list of image paths"""
        self.image_path = path
        for impth in os.listdir(self.image_path):
            path = self.image_path + '/' + impth
            if os.path.isfile(self.image_path + '/' + impth):
                self.image_list.append(path)

    def convert_clahe(self):
        """create clahe folder,  process images with clahe to folder"""
        os.mkdir('clahe_dir')
        os.mkdir('clahe_dir/clahe_dir')
        for file in self.image_list:
            image_name = file.split('/')[-1]
            img = cv2.imread(file, 0)
            create_clahe = cv2.createCLAHE(tileGridSize=(8,8))
            cl1 = create_clahe.apply(img)
            cv2.imwrite('clahe_dir/clahe_dir/' + image_name, cl1)

    def rm_clahe():
        """delete clahe folder"""
        shutil.rmtree('clahe_dir/clahe_dir')
        os.rmdir('clahe_dir')




class EnsembleModel:
    """ get images as input and output ensemble model prediction"""
    def __init__(self):
        self.model_path = 'models'
        self.model_dict = {'vgg': 'models/vgg.pth',
                           'densenet': 'models/densenet.pth',
                           'resnet': 'models/resnext.pth'
                          }

    def prepare_dataset(self, dir):
        """Apply transformations for dataset and get filenames"""
        dataset = ImageFolder(dir,
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
        # arrange to match meta-model
        df_concat = df_concat[['VGG_label_0',
                                 'ResNet_label_0',
                                 'DenseNet_label_0',
                                 'VGG_label_1',
                                 'ResNet_label_1',
                                 'DenseNet_label_1',
                                 'VGG_label_2',
                                 'ResNet_label_2',
                                 'DenseNet_label_2']]
        return df_concat

    def ensemble_prediction(self, df_concat):
        """receive concat dataset output metamodel prediction"""
        with open('models/logistic_regression.pkl', 'rb') as f:
            clf = pickle.load(f)

        preds = clf.predict(df_concat)
        pred_table = pd.DataFrame()
        pred_table['file_name'] =self.file_names
        name_dict = {0:'covid-19', 1:'normal', 2:'pneumonia'}
        pred_table['diagnosis'] = preds
        pred_table['diagnosis'] = pred_table['diagnosis'].map(name_dict)
        return pred_table

    def ensemble_model_predict(self):
        df_concat = self.get_ensemble_table()
        diagnosis = self.ensemble_prediction(df_concat)
        return diagnosis





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

    if os.path.isdir('clahe_dir'):
        Clahe.rm_clahe()


    clahe = Clahe()
    try:
        input = sys.argv[1]
        if os.path.isdir(input):
            clahe.get_image_list(input)
        else:
            print("invalid path provided")
            sys.exit()
    except IndexError:
        print('no path provided initializing demo')
        clahe.get_image_list(DEMO_DIR)



    clahe.convert_clahe()

    master = EnsembleModel()
    master.prepare_dataset('clahe_dir')
    diagnosis = master.ensemble_model_predict()
    print(diagnosis)

    Clahe.rm_clahe()














