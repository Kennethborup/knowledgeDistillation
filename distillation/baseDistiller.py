import torch
import torch.nn as nn
from distillation.utils import Logger
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
                if student is not None:
                    optimizer.load_state_dict(state['optimizer'])
                
        return startEpoch
        
    def init_tensorboard_logger(self, directory=None):
        self.logger = Logger(('logs/' + self.currentTime) if directory is None else directory)
        
    def log(self, epoch, metrics):
        """
        Log performance metrics to tensorboard.
        
        :param epoch: int, current training epoch
        :param metrics: dict, name and metric
        """
        for name, metric in metrics.items():
            self.logger.add_scalar(name, metric, epoch)
        self.logger.flush()
                
    def save(self, epoch, student, teacher, optimizer, directory=None):
        """
        Save checkpoint of model.
        """
        folder = os.path.join('checkpoint', directory if directory is not None else self.currentTime)
        os.makedirs(folder, exist_ok=True)
        torch.save({'epoch': epoch,
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(folder, 'checkpoint.pt'))