'''
The code is to train a huge model, VGG, to test the performance of large model in our own dataset.
The result shows that the performance of VGG is not better as the MobileNetV2
'''

import cv2
from torchvision.models import vgg19_bn
import json
import os
import time
import copy
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR,MultiStepLR


DATA_CLASS_NAMES = {
    "bicycle-lane":0,
    "bicycle-lane-and-pedestrian":1,
    "car-lane":2,
    "pedestrian":3
}

class SequentialDataset(Dataset):
    '''
    generate the sequential image dataset that several images as one input
    '''

    def __init__(self, root_path,  height = 224, width = 224,rescale = None):
        self.root_path = root_path
        self.fnames, self.labels = [], []
        self.height = height
        self.width = width
        self.rescale = rescale
        self.alpha = 0.5
        self.inv = False

        for label in sorted(os.listdir(root_path)):
            for fname in sorted(os.listdir(os.path.join(root_path, label))):
                self.labels.append(DATA_CLASS_NAMES.get(label))
                self.fnames.append(os.path.join(root_path, label, fname))
        assert len(self.labels) == len(self.fnames)


    def __getitem__(self, index):
        frame = np.array(cv2.imread(self.fnames[index])).astype(np.float64)
        labels = np.array(self.labels[index])
        if self.rescale is not None:
            frame = frame*self.rescale
        frame = self.to_tensor(frame)  #更换维度的顺序
        #print('11111111111111111',frame.shape)
        return torch.from_numpy(frame), torch.from_numpy(labels)

    def __len__(self):
        return len(self.fnames)

    def to_tensor(self, buffer):
        return buffer.transpose((2,0,1))


def train_model(log_dir, model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    #use tensorboard
    writer = SummaryWriter(log_dir = log_dir)

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #early_stopping = EarlyStopping(patience=100, verbose=True)
    #scheduler = MultiStepLR(optimizer, [10,50,100,300],0.1)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device,dtype=torch.float)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # require gradients only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss_train = running_loss / len(dataloaders['train'].dataset)
        epoch_acc_train = running_corrects.double() / len(dataloaders['train'].dataset)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss_train, epoch_acc_train))
        writer.add_scalar('train_acc', epoch_acc_train, epoch)
        writer.add_scalar('train_loss', epoch_loss_train, epoch)

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders['eval']:
            inputs = inputs.to(device,dtype=torch.float)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss_eval = running_loss / len(dataloaders['eval'].dataset)
        epoch_acc_eval = running_corrects.double() / len(dataloaders['eval'].dataset)

        if epoch_acc_eval > best_acc:
            best_acc = epoch_acc_eval
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, log_dir + '/checkpoint_best.pkl')

        val_acc_history.append(epoch_acc_eval)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('eval', epoch_loss_eval, epoch_acc_eval))
        writer.add_scalar('valid_acc', epoch_acc_eval, epoch)
        writer.add_scalar('valid_loss', epoch_loss_eval, epoch)
        # save checkpoint
        torch.save(model.state_dict(), log_dir+'/checkpoint.pkl')

        # early stopping
        #early_stopping(epoch_loss_eval, model)
        #if early_stopping.early_stop:
        #    print("Early stopping")
        #    break
        # adjust learning rate
        #scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model, val_acc_history


def main():
    '''
    image = cv2.imread('images/00003.jpg')
    image = maddern2014(image,0.7)
    cv2.imshow('test',image)
    if cv2.waitKey(0) or 0xFF == ord('q'):
        cv2.destroyAllWindows()
    '''
    ## no need when using watcher
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ## todo

    #load parameters
    with open('config/config_torch.json', 'r') as f:
        cfg = json.load(f)
    batch_size = int(cfg['batch'])
    num_epochs = int(cfg['epochs'])
    num_classes = int(cfg['class_number'])
    shape = (int(cfg['height']), int(cfg['width']), 3)
    learning_rate = cfg['learning_rate']
    pre_weights = cfg['weights']
    train_dir = cfg['train_dir']
    eval_dir = cfg['eval_dir']
    test_dir = cfg['test_dir']

    #load model and weights
    model = vgg19_bn(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model = model.cuda()

    #load weights if it has
    if pre_weights and os.path.exists(pre_weights):
        weights = torch.load(pre_weights)
        model.load_state_dict(weights)

    #load dataset
    image_dir = {'train': train_dir,
           'eval': eval_dir}
    dataloaders_dict = {
        x: DataLoader(SequentialDataset(root_path=image_dir[x],rescale=1/255.), batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'eval']}

    #train and evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Send the model to GPU
    model = model.to(device)

    params_to_update = model.parameters()

    #print("Params to learn:")
    #for name, param in model.named_parameters():
    #   if param.requires_grad == True:
    #       print("\t", name)

    # optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9,weight_decay=5e-4)
    optimizer = optim.Adam(params_to_update, lr=learning_rate,weight_decay=1e-5)

    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    log_dir = "train_result/lightweight_network/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    model, hist = train_model(log_dir, model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
    path = log_dir + '/model_final.pkl'
    torch.save(model.state_dict(), path)

    #test the result
    running_loss = 0.0
    running_corrects = 0.0
    test_dataloader = DataLoader(SequentialDataset(root_path= test_dir, rescale=1 / 255.),
                                 batch_size=batch_size, shuffle=True, num_workers=4)

    test_size = len(test_dataloader.dataset)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device,dtype=torch.float)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        probs = torch.nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        running_corrects += torch.sum(preds == labels.data)

    print(test_size)
    print(running_corrects)
    epoch_acc = running_corrects / test_size
    print(epoch_acc)

if __name__ == '__main__':
    main()