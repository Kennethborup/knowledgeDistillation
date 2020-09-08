import torch
import torch.nn as nn
from distillation.utils import Logger, AverageMeter
from datetime import datetime
import os

class BaseTrainer(nn.Module):
    def __init__(self):
        super(BaseTrainer, self).__init__()
        self.currentTime = datetime.now().strftime("%Y%m%d-%H%M%S")
                                
    def print_epoch(self, epoch, epochs, metrics):
        """
        Print training metrics for epoch.
        """
        template = f'Epoch: {epoch:3}/{epochs}'
        for name, metric in metrics.items():
            template += f'\t {name}: {metric:3.3f}'
        print(template)

    def load_state(self, checkpoint, model=None, optimizer=None):
        """
        Load checkpoint if provided and return epoch to start on.
        """
        startEpoch = 1
        if checkpoint:
            if os.path.isfile(checkpoint):
                device = next(model.parameters()).device
                state = torch.load(checkpoint, map_location=device)

                startEpoch = state['epoch']
                if model is not None:
                    model.load_state_dict(state['model'])
                if optimizer is not None:
                    optimizer.load_state_dict(state['optimizer'])
        return startEpoch
        
    def init_tensorboard_logger(self, subDirectory=None):
        self.logDir = os.path.join('logs', '' if subDirectory is None else subDirectory, self.currentTime) 
        self.logger = Logger(self.logDir)
        
    def log(self, epoch, metrics):
        """
        Log performance metrics to tensorboard.
        
        :param epoch: int, current training epoch
        :param metrics: dict, name and metric
        """
        for name, metric in metrics.items():
            self.logger.add_scalar(name, metric, epoch)
        self.logger.flush()
                
    def save(self, epoch, model, optimizer, metrics, subDirectory=None):
        """
        Save checkpoint of model.
        """
        self.checkpointDir = os.path.join('checkpoint', '' if subDirectory is None else subDirectory, self.currentTime)
        os.makedirs(self.checkpointDir, exist_ok=True)
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics': metrics},
                   os.path.join(self.checkpointDir, 'checkpoint.pt'))
        
    def validate(self, model, dataloader, objective, metric):
        """
        Validate model on all data in dataloader.
        
        :return: dict, named metrics for logging.
        """
        model.eval()
        
        device = next(model.parameters()).device
        lossMeter = AverageMeter()
        metricMeter = AverageMeter()
        
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                # Calculate logits
                logits = model(data)

                # Calculate loss
                batchLoss = objective(logits, target)
            
            # Save metrics
            lossMeter.update(batchLoss.item(), n=len(data))
            metricMeter.update(metric(logits, target), n=len(data))
        
        return {'Valid/Loss': lossMeter.avg,
                'Valid/Metric': metricMeter.avg}
    
    def train_step(self, model, dataloader, objective, metric, optimizer):
        """
        Train model on data from dataloader for one epoch.
        
        :return: dict, named metrics for logging.
        """
        model.train()
        
        device = next(model.parameters()).device
        lossMeter = AverageMeter()
        metricMeter = AverageMeter()
        
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Calculate logits
            logits = model(data)
            
            # Calculate loss
            optimizer.zero_grad()
            batchLoss = objective(logits, target)

            # Update model weights
            batchLoss.backward()
            optimizer.step()
            
            # Save metrics
            lossMeter.update(batchLoss.item(), n=len(data))
            metricMeter.update(metric(logits, target), n=len(data))
        
        return {'Train/Loss': lossMeter.avg,
                'Train/Metric': metricMeter.avg}