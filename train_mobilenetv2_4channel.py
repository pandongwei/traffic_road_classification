'''
this program is to train the mobilenetv2 with 4 channels' input, in order to eliminate the influence of strong light condition
'''

import cv2
from model.mobilenetv2_4channels import mobilenetv2_4channels
import json
import os
import time
import copy
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader,Dataset
from kalman_filter import *
from utils import errorCalculation_return_missclassList

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
        tmp = frame.copy()
        if self.rescale is not None:
            frame = frame*self.rescale
        extra_channel = maddern2014(tmp,alpha=self.alpha,inv=self.inv) # rescale is already included in this function
        frame = np.concatenate((frame,extra_channel),axis=2)
        frame = self.to_tensor(frame)  #更换维度的顺序
        return torch.from_numpy(frame), torch.from_numpy(labels)

    def __len__(self):
        return len(self.fnames)

    def to_tensor(self, buffer):
        return buffer.transpose((2,0,1))


def maddern2014(image,alpha,inv=False):
    """
    Implement the algorithm proposed by Will Maddern et al. in ICRA2014
    Paper:
    Illumination Invariant Imaging: Applications in Robust Vision-based
    Localisation, Mapping and Classification for Autonomous Vehicles

    ii_image = rgb2ii.maddern2014(image, alpha, inv)
    where
    image : color image data
    alpha : a camera-dependent parameter ranged in 0-1
    inv   : perform image inversion (a=1-a) if inv is true (default: false)
    """
    image = image / 255.
    #ii_image = 0.5 + math.log(image[:,:,1]) - alpha*math.log(image[:,:,2]) - (1-alpha)*math.log(image[:,:,0])
    ii_image = np.full((224,224),0.5) + np.log(image[:, :, 1] + 1e-10) - alpha * np.log(image[:, :, 2] + 1e-10) - (1 - alpha) * np.log(
        image[:, :, 0] + 1e-10)
    if inv:
        ii_image = 1-ii_image
    return ii_image[:,:,np.newaxis]

def train_model(log_dir, model, dataloaders, criterion, optimizer, num_epochs,batch_size):
    since = time.time()

    #use tensorboard
    writer = SummaryWriter(log_dir = log_dir)

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device,dtype=torch.float)
                labels = labels.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                #zeros = torch.zeros(inputs.size(0),model.last_channel,
                #                           dtype=inputs.dtype, device=inputs.device)

                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)


                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model,这里只保存了最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train:':
                epoch_acc_train = epoch_acc
                epoch_loss_train = epoch_loss
                writer.add_scalar('train_acc', epoch_acc_train, epoch)
                writer.add_scalar('train_loss', epoch_loss_train, epoch)
            if phase == 'val':
                epoch_acc_eval = epoch_acc
                epoch_loss_eval = epoch_loss
                writer.add_scalar('valid_acc', epoch_acc_eval, epoch)
                writer.add_scalar('valid_loss', epoch_loss_eval, epoch)
        torch.save(model.state_dict(), log_dir+'/checkpoint.pkl')
        print()

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
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"  ## todo

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
    model = mobilenetv2_4channels(pretrained=True,n_class=num_classes)
    model = model.cuda()

    #load weights if it has
    if pre_weights and os.path.exists(pre_weights):
        weights = torch.load(pre_weights)
        model.load_state_dict(weights)

    #load dataset
    image_dir = {'train': train_dir,
           'val': eval_dir}
    dataloaders_dict = {
        x: DataLoader(SequentialDataset(root_path=image_dir[x],rescale=1/255.), batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

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
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    log_dir = "train_result/lightweight_network/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    model, hist = train_model(log_dir, model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, batch_size=batch_size)
    path = log_dir + '/model_final.pkl'
    torch.save(model.state_dict(), path)

    #test the result
    running_loss = 0.0
    running_corrects = 0.0
    test_dataloader = DataLoader(SequentialDataset(root_path= test_dir, rescale=1 / 255.),
                                 batch_size=batch_size, num_workers=4)

    test_size = len(test_dataloader.dataset)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    class_label = []
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device,dtype=torch.float)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        probs = torch.nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        running_corrects += torch.sum(preds == labels.data)
        predictions.extend(outputs.cpu().numpy())
        class_label.extend(labels.cpu().numpy())

    print(test_size)
    print(running_corrects)
    epoch_acc = running_corrects / test_size
    print(epoch_acc)

    # combine kalman filter
    predictions = np.asarray(predictions)
    print(predictions.shape)
    predictions = predictions.reshape((-1, 4))
    print(predictions.shape)
    predict_prob = predictions.T # (4,1035)
    output_kalman = []
    x_state = KalmanFilter()
    for i in range(len(predict_prob[0])):
        # kalman part
        x_state.predict()
        x_state.correct(predict_prob[:, i])
        x_combination = x_state.x_e
        output_kalman.append(np.argmax(x_combination))
    error_kalman, missclass_list_kalman = errorCalculation_return_missclassList(class_label, output_kalman)
    print("after combining kalman filter the classifier got {}% of the testing examples correct!".format(
        round((1.0 - error_kalman) * 100, 2)))

if __name__ == '__main__':
    main()