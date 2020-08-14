import torch
import torch.nn as nn
from distillation.utils import Logger, AverageMeter, Accuracy
from datetime import datetime
import os


class BaseDistiller(nn.Module):
    def __init__(self):
        super(BaseDistiller, self).__init__()
        self.currentTime = datetime.now().strftime("%Y%m%d-%H%M%S")
        
    def _setHook(self, hook, model, layer):
        """
        Set hook for distillation on provided model and layer.
        """
        hook.setHook(self._getLayer(model, layer))
    
    def _getLayer(self, model, layer):
        """
        Fetch layer from model.
        
        :param layer: int or str; layer position or layer name for backbone model, to use as distillation layer
        """
        if type(layer) == str:
            modules = dict([*model.named_modules()])
            return modules.get(layer, None)
        elif type(layer) == int:
            children = [*model.children()]
            return children[layer]
        else:
            raise  NameError(f'Hidden layer ({layer}) not found in model!')
                        
    def print_epoch(self, epoch, epochs, metrics):
        """
        Print training metrics for epoch.
        """
        template = f'Epoch: {epoch:3}/{epochs}'
        for name, metric in metrics.items():
            template += f'\t {name}: {metric:3.3f}'
        print(template)

    def load_state(self, checkpoint, student=None, teacher=None, optimizer=None):
        """
        Load checkpoint if provided and return epoch to start on.
        """
        startEpoch = 1
        if checkpoint:
            if os.path.isfile(checkpoint):
                device = next(student.parameters()).device
                state = torch.load(checkpoint, map_location=device)

                startEpoch = state['epoch']
                if student is not None:
                    student.load_state_dict(state['student'])
                if teacher is not None:
                    teacher.load_state_dict(state['teacher'])
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
                
    def save(self, epoch, student, teacher, optimizer, subDirectory=None):
        """
        Save checkpoint of model.
        """
        self.checkpointDir = os.path.join('checkpoint', '' if subDirectory is None else subDirectory, self.currentTime)
        os.makedirs(self.checkpointDir, exist_ok=True)
        torch.save({'epoch': epoch,
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(self.checkpointDir, 'checkpoint.pt'))
        
    def validate(self, student, dataloader, objective, OneHot=False):
        """
        Validate student model on all data in dataloader.
        
        :return: dict, named metrics for logging.
        """
        student.eval()
        
        device = next(student.parameters()).device
        accuracy = Accuracy(OH=OneHot)
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                # Calculate logits
                sLogits = student(data)

                # Calculate loss
                batchLoss = objective(nn.functional.log_softmax(sLogits, dim=1), target)
            
            # Save metrics
            lossMeter.update(batchLoss.item(), n=len(data))
            accMeter.update(accuracy(nn.functional.softmax(sLogits, dim=1), target), n=len(data))
        
        return {'Valid/Loss': lossMeter.avg,
                'Valid/Accuracy': accMeter.avg}