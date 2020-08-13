import torch
import torch.nn as nn
from torch.utils.tensorboard.summary import hparams

#####################################
# Misc.
#####################################
class Hook():
    """
    A simple hook class that returns the output of a layer of a model during forward pass.
    """
    def __init__(self):
        self.output = None
        
        
    def setHook(self, module):
        """
        Attaches hook to model.
        """
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """
        Saves the wanted information.
        """
        self.output = output
        
    def val(self):
        """
        Return the saved value.
        """
        return self.output

    def hooked(self):
        """
        Returns True if setHook has been called, else False.
        """
        return hasattr(self, 'hook')


#####################################
# Updated tensorboard logger
#####################################
class Logger(torch.utils.tensorboard.SummaryWriter):    
    def add_hparams(self, hparam_dict, metric_dict):
        """Alteration to the offical SummaryWriter from PyTorch, which creates
        a new tensorboard event file with the hyperparameters and adds additional
        scalars to the scalar-tab with the registered metric value.
        
        This is unfortunate behavior, and the below merely adds the hyperparameters
        to the existing eventfile.
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self._get_file_writer().add_summary(exp)
        self._get_file_writer().add_summary(ssi)
        self._get_file_writer().add_summary(sei)
        

#####################################
# Running metric
#####################################
class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
#####################################
# Metrics
#####################################
class Accuracy(nn.Module):
    def __init__(self, OH=False):
        super(Accuracy).__init__()
        self.OH = OH
        
    def __call__(self, pred, target):
        if self.OH:
            target = torch.argmax(target, dim=1).int()
        pred = torch.argmax(pred, dim=1).int()
        
        return pred.eq(target.view_as(pred)).float().mean().item()

    
#####################################
# For Example
#####################################  
class MLP(nn.Module):
    """
    Simple MLP model for projector and predictor in BYOL paper.
    
    :param inputDim: int; amount of input nodes
    :param projectionDim: int; amount of output nodes
    :param hiddenDim: int; amount of hidden nodes
    """
    def __init__(self, inputDim, projectionDim, hiddenDim=4096):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(inputDim, hiddenDim)
        self.bn = nn.BatchNorm1d(hiddenDim)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hiddenDim, projectionDim)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
    
class PseudoDataset(torch.utils.data.Dataset):
    """
    Pseudo dataset producing random batches with some specified data shape and 10 class output.
    """
    def __init__(self, size):
        self.size = size
      
    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return torch.rand(self.size), torch.randint(0,10,(1,))