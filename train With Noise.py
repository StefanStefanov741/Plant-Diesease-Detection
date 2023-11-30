import os                       #for working with files
import numpy as np              #for numerical computationss
import pandas as pd             #for working with dataframes
import torch                    #Pytorch module 
import matplotlib.pyplot as plt #for plotting informations on graph and images using tensors
import torch.nn as nn           #for creating  neural networks
from torch.utils.data import DataLoader, Dataset #for dataloaders 
from PIL import Image           #for checking images
import torch.nn.functional as F #for functions for calculating loss
import torchvision.transforms as transforms   #for transforming images into tensors 
from torchvision.utils import make_grid       #for data checking
from torchvision.datasets import ImageFolder  #for working with classes and images
from torchsummary import summary              #for getting the summary of our model
from ResNet import ResNet9

if __name__ == '__main__':

    data_dir = "C:/Users/tetij/Desktop/IVP/Plant Dataset 4GB"
    train_dir = os.path.join(data_dir, "trainWithNoise")
    valid_dir = os.path.join(data_dir, "valid")
    diseases = os.listdir(train_dir)

    #for getting available device (GPU if possible)
    def get_default_device():
        if torch.cuda.is_available:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    #for moving data to device (CPU or GPU)
    def to_device(data, device):
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    #for loading in the device (GPU if available else CPU)
    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)
            
        def __len__(self):
            """Number of batches"""
            return len(self.dl)
        
    def plot_accuracies(history):
        accuracies = [x['val_accuracy'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')

    def plot_losses(history):
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        
    def plot_lrs(history):
        lrs = np.concatenate([x.get('lrs', []) for x in history])
        plt.plot(lrs)
        plt.xlabel('Batch no.')
        plt.ylabel('Learning rate')
        plt.title('Learning Rate vs. Batch no.')

    #datasets for training and validation
    train = ImageFolder(train_dir, transform=transforms.ToTensor())
    valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

    #Setting the seed value
    random_seed = 42
    torch.manual_seed(random_seed)

    #setting the batch size
    batch_size = 32

    #DataLoaders for training and validation
    train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dl = DataLoader(valid, batch_size, num_workers=4, pin_memory=True)

    device = get_default_device()

    #Moving data into GPU
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    model = to_device(ResNet9(3, len(train.classes)), device)

    #print(model)

    #getting summary of the model
    INPUT_SHAPE = (3, 256, 256)

    #for training
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)


    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        

    def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                    grad_clip=None, opt_func=torch.optim.SGD):
        torch.cuda.empty_cache()
        history = []
        
        optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        #scheduler for one cycle learniing rate
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

        
        for epoch in range(epochs):
            #Training
            model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                
                #gradient clipping
                if grad_clip: 
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                    
                optimizer.step()
                optimizer.zero_grad()
                
                #recording and updating learning rates
                lrs.append(get_lr(optimizer))
                sched.step()
                
        
            #validation
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            model.epoch_end(epoch, result)
            history.append(result)
            
        return history

    history = [evaluate(model, valid_dl)]
    #print(history)

    #Second change: increase of epochs
    epochs = 2
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    history += fit_OneCycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=1e-4, 
                             opt_func=opt_func)

    #saving the entire model to working directory
    PATH = './plant-disease-model-augmentation-PlusNoise.pth'
    torch.save(model, PATH)