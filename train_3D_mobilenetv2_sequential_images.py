'''
This code is to train and test the 3D_MobileNetV2 model with/without LSTM
PS: In the model the train and test dataset must be continuous, to keep the temporal relationship.
The input is the image sequence, you can set up the length of image sequence in config file
'''
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
from model import mobilenetv2_3D,mobilenetv2_3D_LSTM
import json
import time
import copy
from datetime import datetime
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

DATA_CLASS_NAMES = {
    "bicycle-lane":0,
    "bicycle-lane-and-pedestrian":1,
    "car-lane":2,
    "pedestrian":3
}

class SequentialDataset_2(Dataset):
    '''
    generate the sequential image dataset that several images as one input
    '''

    def __init__(self, root_path, images_len=10,  height=224, width=224, rescale=None):
        self.root_path = root_path
        self.images_len = images_len
        self.fnames, self.labels = [], []
        self.height = height
        self.width = width
        self.rescale = rescale
        for label in sorted(os.listdir(root_path)):
            i = 0
            images_path = sorted(os.listdir(os.path.join(root_path, label)))
            l = len(images_path)
            while i <= l - self.images_len:
                part = []
                for j in range(self.images_len):
                    part.append(os.path.join(root_path, label, images_path[i+j]))

                self.labels.append(DATA_CLASS_NAMES.get(label))
                self.fnames.append(part)
                i += 5
        assert len(self.labels) == len(self.fnames)


    def __getitem__(self, index):
        buffer = np.empty((self.images_len, self.height, self.width, 3), np.dtype('float32'))
        for i,frame_name in enumerate(self.fnames[index]):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            if i < self.images_len:
                buffer[i] = frame
            else:
                break
        labels = np.array(self.labels[index])
        if self.rescale is not None:
            buffer = buffer*self.rescale
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def __len__(self):
        return len(self.fnames)

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

class SequentialDataset(Dataset):
    '''
    generate the sequential image dataset that several images as one input
    '''

    def __init__(self, root_path, images_len=10,  height=224, width=224, rescale=None):
        self.root_path = root_path
        self.images_len = images_len
        self.fnames, self.labels = [], []
        self.height = height
        self.width = width
        self.rescale = rescale
        part = []
        for label in sorted(os.listdir(root_path)):
            i = 0
            for fname in sorted(os.listdir(os.path.join(root_path, label))):
                if i < self.images_len:
                    part.append(os.path.join(root_path, label, fname))
                    i += 1
                else:
                    self.labels.append(DATA_CLASS_NAMES.get(label))
                    self.fnames.append(part)
                    part = []
                    i = 0
        assert len(self.labels) == len(self.fnames)


    def __getitem__(self, index):
        buffer = np.empty((self.images_len, self.height, self.width, 3), np.dtype('float32'))
        for i,frame_name in enumerate(self.fnames[index]):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            if i < self.images_len:
                buffer[i] = frame
            else:
                break
        labels = np.array(self.labels[index])
        if self.rescale is not None:
            buffer = buffer*self.rescale
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def __len__(self):
        return len(self.fnames)

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def train_model(log_dir, model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    #use tensorboard
    writer = SummaryWriter(log_dir = log_dir)

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(patience=100, verbose=True)
    #scheduler = CosineAnnealingLR(optimizer, num_epochs)
    scheduler = MultiStepLR(optimizer,[30,60],gamma=0.1,last_epoch=-1)
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
            inputs = inputs.to(device)
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
            inputs = inputs.to(device)
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
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model, val_acc_history

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ## todo

    with open('config/config_3dcnn.json', 'r') as f:
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
    model = cfg['model']
    images_len = cfg['images_len']
    #选择模型 (3D-MobileNetV2,3D-MobileNetV2+LSTM)
    if model == '3D-MobileNetV2':
        model = mobilenetv2_3D.get_model(num_classes=num_classes, sample_size=shape[0], width_mult=1.0)
    elif model == '3D-MobileNetV2-LSTM':
        model = mobilenetv2_3D_LSTM.get_model(num_classes=num_classes, sample_size=shape[0], width_mult=1.0)


    model = model.cuda()

    #load weights if it has
    if pre_weights and os.path.exists(pre_weights):
        weights = torch.load(pre_weights)
        model.load_state_dict(weights)

    '''
    model_path = r'./logs/fit/1/jester_mobilenetv2_1.0x_RGB_16_best.pth'
    checkpoint = torch.load(model_path)
    #load part of the weight because of finetune
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    '''

    image_dir = {'train': train_dir,
           'eval': eval_dir}
    #dataloaders_dict = {
    #    x: DataLoader(SequentialDataset_2(root_path=image_dir[x],images_len=16,rescale=1/255.), batch_size=batch_size, shuffle= True, num_workers=4) for x in
    #    ['train', 'eval']}
    dataloaders_dict = {
        'train': DataLoader(SequentialDataset_2(root_path=image_dir['train'],images_len=images_len,rescale=1/255.), batch_size=batch_size, shuffle= True, num_workers=4),
        'eval': DataLoader(SequentialDataset_2(root_path=image_dir['eval'],images_len=images_len,rescale=1/255.), batch_size=batch_size, shuffle= True, num_workers=4)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Send the model to GPU
    model = model.to(device)

    params_to_update = model.parameters()

    #print("Params to learn:")
    #for name, param in model.named_parameters():
    #    if param.requires_grad == True:
    #        print("\t", name)

    # Observe that all parameters are being optimized
    #optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9,weight_decay=5e-4)
    optimizer = optim.Adam(params_to_update,lr=learning_rate,weight_decay=4e-4)

    # Setup the loss fxn
    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    log_dir = "train_result/lightweight_network/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    model, hist = train_model(log_dir, model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
    path = log_dir + '/model_final.pkl'
    torch.save(model.state_dict(), path)

    #test the result
    running_loss = 0.0
    running_corrects = 0.0
    test_dataloader = DataLoader(SequentialDataset(root_path= test_dir, images_len=images_len, rescale=1 / 255.),
                                 batch_size=batch_size, shuffle=True ,num_workers=4)

    test_size = len(test_dataloader.dataset)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
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


if __name__ == "__main__":
    main()
