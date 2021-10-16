import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelBase(nn.Module):

    # this is for loading the batch of train image and outputting its loss, accuracy
    # & predictions
    def training_step(self, batch, weight):
        images, labels = batch
        out = self(images)  # generate predictions
        # weighted compute loss
        loss = F.cross_entropy(out, labels, weight=weight)
        acc, preds = accuracy(out, labels)  # calculate accuracy

        return {'train_loss': loss, 'train_acc': acc}

    # this is for computing the train average loss and acc for each epoch
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss']
                        for x in outputs]  # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses
        batch_accs = [x['train_acc']
                      for x in outputs]  # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()  # combine accuracies

        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}

    def predict(self, batch):
        images, labels = batch
        out = self(images)  # generate predictions
        return out

    # this is for loading the batch of val/test image and outputting its loss, accuracy,
    # predictions & labels
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # generate predictions
        loss = F.cross_entropy(out, labels)  # compute loss
        acc, preds = accuracy(out, labels)  # calculate acc & get preds

        return {'val_loss': loss.detach(), 'val_acc': acc.detach(),
                'preds': preds.detach(), 'labels': labels.detach(),
                'out': out}  # IMPORTANT added out

    # detach extracts only the needed number, or other numbers will crowd memory

    # this is for computing the validation average loss and acc for each epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss']
                        for x in outputs]  # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses
        batch_accs = [x['val_acc'] for x in outputs]  # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()  # combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # this is for printing out the results after each epoch
    def epoch_end(self, epoch, train_result, val_result):
        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.
              format(epoch + 1, train_result['train_loss'], train_result['train_acc'],
                     val_result['val_loss'], val_result['val_acc']))

    # this is for using on the test set, it outputs the average loss and acc,
    # and outputs the predictions
    def test_prediction(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # combine accuracies
        # combine predictions
        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()]
        # combine labels
        batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]

        # IMPORTANT added out
        out = [x['out'] for x in outputs]

        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(),
                'test_preds': batch_preds, 'test_labels': batch_labels, 'out': out}


class ModelNet(ModelBase):
    def __init__(self, network):
        super().__init__()
        # Use a pretrained model
        self.network = network
        # Freeze training for all layers before classifier
        # resnet
        if self.network.__class__.__name__ == 'ResNet':
            for param in self.network.fc.parameters():
                param.require_grad = False
        # vgg16
        if self.network.__class__.__name__ == 'VGG':
            for param in self.network.classifier[6].parameters():
                param.require_grad = False
        # densenet
        if self.network.__class__.__name__ == 'DenseNet':
            for param in self.network.classifier.parameters():
                param.require_grad = False

    def forward(self, xb):
        return self.network(xb)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds


def F1_score(outputs, labels):
    _, preds = torch.max(outputs, dim=1)

    # precision, recall, and F1
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1, preds
