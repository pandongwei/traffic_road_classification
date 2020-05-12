import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
#from mobilenet_v2_lstmcell_torch import *
from model.mobilenetv2_LSTM import *
import json
import time
import copy
from datetime import datetime
import cv2


def train_model(log_dir, model, dataloaders, criterion, optimizer, num_epochs,batch_size):
    since = time.time()

    #use tensorboard
    writer = SummaryWriter(log_dir = log_dir)
    #dummy_input = torch.rand(16, 3, 224, 224, device=torch.device('cuda'))
    #writer.add_graph(model, dummy_input)
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

            zeros = torch.zeros(1, batch_size, 1280,
                                dtype=torch.float32, device=device)
            hidden_state = (zeros, zeros)  #TODO

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if len(inputs) < batch_size: break  #TODO
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
                    outputs, hidden_state = model(inputs,hidden_state)

                    loss = criterion(outputs, labels)
                    #for i in hidden_state: i.detach_()  #TODO

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()

                    outputs, hidden_state = model(inputs,hidden_state)

                    loss = criterion(outputs, labels)
                    for i in hidden_state: i.detach_()
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
    #set up gpu usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ## todo

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
    model = mobilenetv2_lstmcell(pretrained=True,n_class=num_classes)
    model = model.cuda()

    #load weights if it has
    if pre_weights and os.path.exists(pre_weights):
        weights = torch.load(pre_weights)
        model.load_state_dict(weights)

    #load dataset
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [255, 255, 255])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [255, 255, 255])
        ]),
    }
    data_dir = {'train': train_dir,
                 'val': eval_dir}
    image_datasets = {x: ImageFolder(data_dir[x], data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in
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
    model, hist = train_model(log_dir, model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs,batch_size=batch_size)
    path = log_dir + '/model_final.pkl'
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()