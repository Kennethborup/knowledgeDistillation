import torch
import torch.nn as nn
from distillation.utils import Accuracy, AverageMeter, Hook
from distillation.baseDistiller import BaseDistiller

class AttentionDistiller(BaseDistiller):
    def __init__(self, alpha, studentLayer=[-2], teacherLayer=[-2], p=2):
        super(AttentionDistiller, self).__init__()
        assert len(studentLayer) == len(teacherLayer), "You must provide the same amount of student and teacher layers."
        
        self.alpha = alpha
        self.p = p
        self.studentLayer = studentLayer
        self.teacherLayer = teacherLayer
        self.layers = len(studentLayer)
        
        # Register hooks
        self.studentHooks =[Hook() for _ in range(self.layers)]
        self.teacherHooks = [Hook() for _ in range(self.layers)]
    
    def _attention(self, x, p=2):
        """
        Takes B x C x H x W tensors and returns the attention summed across the channels (B x H x W)
        """
        return nn.functional.normalize(x.pow(p).sum(dim=-1), dim=-1).flatten()


    def _attentionLoss(self, studentActivations, teacherActivations, p=2):
        """
        Calculates the attention loss between the teacher and student activations according to https://arxiv.org/abs/1612.03928
        """
        studentQ = [self._attention(act) for act in studentActivations]
        teacherQ = [self._attention(act) for act in teacherActivations]
        normDiff = [(sQ - tQ).norm(p) for sQ, tQ in zip(studentQ, teacherQ)]
        return sum(normDiff)

    def train_step(self, student, teacher, dataloader, optimizer, objective, distillObjective=None, OneHot=False):
        """
        Train student model to the teacher model for one epoch with Attention distillation (https://arxiv.org/abs/1612.03928).
        Note the distillObjective is ignored, as it is pre-specified.
        
        :return: dict, named metrics for logging.
        """
        student.train()
        teacher.eval()
        
        # Attach
        if not self.studentHooks[0].hooked():
            for hook, layer in zip(self.studentHooks, self.studentLayer):
                self._setHook(hook, student, layer)
        if not self.teacherHooks[0].hooked():
            for hook, layer in zip(self.teacherHooks, self.teacherLayer):
                self._setHook(hook, teacher, layer)

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
            sAct = [hook.val() for hook in self.studentHooks]
            tAct = [hook.val() for hook in self.teacherHooks]
            
            # Calculate loss
            optimizer.zero_grad()
            batchLoss = (1-self.alpha)*self._attentionLoss(sAct, tAct, p=self.p)
            batchLoss += self.alpha*objective(nn.functional.log_softmax(sLogits, dim=1), target)

            # Update student weights
            batchLoss.backward()
            optimizer.step()
            
            # Save metrics
            lossMeter.update(batchLoss.item(), n=len(data))
            accMeter.update(accuracy(nn.functional.softmax(sLogits, dim=1), target), n=len(data))
        
        return {'Train/Loss': lossMeter.avg,
                'Train/Accuracy': accMeter.avg}
