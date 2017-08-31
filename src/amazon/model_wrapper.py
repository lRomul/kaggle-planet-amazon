import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .utils import AverageMeter
from .loggers import PrintLogger


class Model:
    
    def __init__(self, model, criterion, optimizer,  logger=PrintLogger()):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.epoch = 0
        self.best_loss = np.inf
        self.state_path=''

        self.logger = logger

    def _run_epoch(self, data_loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        loss_meter = AverageMeter()

        for i, (input, target) in enumerate(data_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=not train)
            target_var = torch.autograd.Variable(target, volatile=not train)

            if train:
                self.optimizer.zero_grad()

            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            if train:
                loss.backward()
                self.optimizer.step()

            loss_meter.update(loss.data[0], input.size()[0])

        return loss_meter.avg

    def validate(self, val_loader):
        return self._run_epoch(val_loader, train=False)

    def fit(self, train_loader, val_loader, n_epoch=1, lr=0.01):
        history = {stat:[] for stat in ['train', 'val']}
        val_loss = self.validate(val_loader)

        message = "Epoch {0}: train {1} \t val {2}"
        self.logger.log("Val: {0}".format(val_loss))
        for epoch in range(n_epoch):
            self.adjust_learning_rate(lr, epoch)

            train_loss = self._run_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.logger.log(message.format(self.epoch, train_loss, val_loss))
            self.epoch += 1
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model(self.state_path)
            
            history['train'].append(train_loss)
            history['val'].append(val_loss)
        return history
            
    def set_savestate_path(self, path):
        self.state_path = path
        
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate(self, base_lr, epoch):
        lr = base_lr * (0.33 ** (epoch / 30))
        self.set_lr(lr)
            
    def save_model(self, filename):
        if filename:
            state = {
                'state_dict': self.model.state_dict(),
                'epoch': self.epoch,
                'best_loss': self.best_loss
            }
            torch.save(state, filename)
            self.logger.log("Model saved")
        
    def load_model(self, filename):
        if os.path.isfile(filename):
            state = torch.load(filename)
            self.model.load_state_dict(state['state_dict'])
            self.epoch = state['epoch']
            self.best_loss = state['best_loss']
        else:
            raise Exception("No state found at {}".format(filename))
            
    def __call__(self, x):
        self.model.eval()
        return F.sigmoid(self.model(x))
    
    def predict(self, tensor):
        self.model.eval()
        one_image = len(tensor.size()) == 3
        if one_image:
            tensor = tensor.unsqueeze(0)
        inputs = Variable(tensor.cuda())
        output = F.sigmoid(self.model(inputs))
        if one_image:
            output = output.cpu().data[0]
        else:
            output = output.cpu().data
        return output