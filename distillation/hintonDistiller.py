import torch
import torch.nn as nn
from distillation.utils import Accuracy, AverageMeter, Hook
from distillation.baseDistiller import BaseDistiller
import time

class HintonDistiller(BaseDistiller):
    def __init__(self, alpha, studentLayer=-2, teacherLayer=-2):
        super(HintonDistiller, self).__init__()
        
        self.alpha = alpha
        self.studentLayer = studentLayer
        self.teacherLayer = teacherLayer
        
        # Register hooks
        self.studentHook = Hook()
        self.teacherHook = Hook()

    def train_step(self, student, teacher, dataloader, objective, distillObjective, optimizer, OneHot=False):
        """
        Train student model to the teacher model for one epoch with Hinton KD.
        
        :return: dict, named metrics for logging.
        """
        student.train()
        teacher.eval()
        
        # Attach
        if not self.studentHook.hooked():
            self._setHook(self.studentHook, student, self.studentLayer)
        if not self.teacherHook.hooked():
            self._setHook(self.teacherHook, teacher, self.teacherLayer)

        device = next(student.parameters()).device
        accuracy = Accuracy(OH=OneHot)
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Calculate logits
            sLogits = student(data)
            tLogits = teacher(data)
            
            # Retrieve activations from distillation layer of both models
            sAct = self.studentHook.val()
            tAct = self.teacherHook.val()
            
            # Calculate loss
            optimizer.zero_grad()
            batchLoss = (1-self.alpha)*distillObjective(nn.functional.log_softmax(sAct, dim=1), nn.functional.softmax(tAct, dim=1))
            batchLoss += self.alpha*objective(nn.functional.log_softmax(sLogits, dim=1), target)

            # Update student weights
            batchLoss.backward()
            optimizer.step()
            
            # Save metrics
            lossMeter.update(batchLoss.item(), n=len(data))
            accMeter.update(accuracy(nn.functional.softmax(sLogits, dim=1), target), n=len(data))
        
        return {'Train/Loss': lossMeter.avg,
                'Train/Accuracy': accMeter.avg}
