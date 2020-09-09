import torch
import torch.nn as nn
from torch.utils.tensorboard.summary import hparams
import numpy as np

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
# Teacher weight annealing
#####################################
class TeacherAnnealing(object):
    """
    Updates the (1-alpha)-weight for knowledge distillation according to CosineAnnealing
    \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} -
        \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
    """
    def __init__(self, distiller, endVal, epochs):
        """
        Args:
            distiller: distiller to update. Must have alpha attribute
            endVal: val to anneal to in èpochs` iterations
            epochs: amouint of iterations to reach `endVal`
        """
        self.distiller = distiller
        self.etaMax = self.distiller.alpha
        self.etaMin = endVal
        self.tMax = epochs
        self.tCur = 0
        
    def reset(self):
        self.tCur = 0
        self.distiller.alpha = self.etaMax
    
    def step(self):
        self.tCur += 1
        self.distiller.alpha = 1-(self.etaMin + 0.5*(self.etaMax - self.etaMin)*(1 + torch.cos(self.tCur/self.tMax * torch.tensor(np.pi))))
        
    def get_val(self):
        return self.distiller.alpha

    
#####################################
# Custom loss function
#####################################
class FocalKLD(object):
    """Focal-like weighting of KLD"""
    def __call__(self, yTrue, yStudent, yTeacher):
        weights = yTeacher[torch.arange(yTeacher.size(0)), yTrue]
        kld = torch.nn.functional.kl_div(yTeacher, yStudent, reduction='none')
        return torch.sum(weights.unsqueeze(1)*kld, axis=1).mean()
    
    
#####################################
# For Examples
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

    
class CNN(nn.Module):
    """
    Simple CNN model for data free classification example.
    
    :param inputSize: tuple; amount of input nodes
    :param projectionDim: int; amount of output nodes
    :param hiddenDim: int; amount of hidden nodes
    """
    def __init__(self, inputSize, projectionDim, hiddenDim=32):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inputSize[0], out_channels=hiddenDim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(hiddenDim)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(inputSize[1]*inputSize[2]*hiddenDim, projectionDim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.l2(x)
        return x
    
    
class Generator(nn.Module):
    """
    Simple Generator model.
    
    :param inputDim: int; amount of input nodes
    :param imgSize: tuple; size of output images
    """
    def __init__(self, inputDim=100, imgSize=(1, 32, 32)):
        super(Generator, self).__init__()
        self.outputDims = (imgSize[0], imgSize[1]//4, imgSize[2]//4)
        
        self.l1 = nn.Sequential(
            nn.Linear(inputDim, 128*self.outputDims[1]*self.outputDims[2])
        )

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.outputDims[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(self.outputDims[0], affine=False) 
        )

    def forward(self, z):
        img = self.l1(z.view(z.size(0), -1))
        img = img.view(img.size(0), -1, self.outputDims[1], self.outputDims[2])
        img = self.conv_blocks0(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img
    
    
class PseudoDataset(torch.utils.data.Dataset):
    """
    Pseudo dataset producing random batches with some specified data shape and 10 class output.
    """
    def __init__(self, size):
        self.size = size
      
    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return torch.rand(self.size), torch.randint(0,10,())
    
    
# Scalewrapper
class ScaleWrapper(nn.Module):
    def __init__(self, interval):
        super(ScaleWrapper, self).__init__()
        
    def _scaler(self, x):
        raise NotImplementedError('_scaler should be implemented in descendent of ScaleWrap class!')
        
    def __call__(self, x):        
        # Apply scaler and return to interval scale.
        return (self.interval[1]-self.interval[0])*self._scaler(x) + self.interval[0]
        
class SigmoidScaler(ScaleWrapper):
    def __init__(self, interval, p=2.463):
        super(SigmoidScaler, self).__init__(interval)
        self.p = p
        self.interval = interval
        self.scale = (self.interval[1] - self.interval[0])/2
        self.center = (self.interval[0] + self.interval[1])/2
        
    def _scaler(self, x):
        return torch.sigmoid(self.p/self.scale * (x - self.center))