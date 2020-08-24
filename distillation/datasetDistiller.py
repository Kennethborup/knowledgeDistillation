import torch
import torch.nn as nn
from distillation.utils import Accuracy, AverageMeter, Hook
from distillation.baseDistiller import BaseDistiller
import random
import os

class DatasetDistiller(BaseDistiller):
    def __init__(self, pseudoIters, studentIters, pseudoLR, scaler, batchSize=64, pseudoSize=(3, 32,32)):
        super(DatasetDistiller, self).__init__()
        # Training scheme
        self.studentIters = studentIters
        self.pseudoIters = pseudoIters
        self.epochIters = self.studentIters + self.pseudoIters
        
        # pseudoData
        self.scaler = scaler
        self.pseudoData = torch.rand((batchSize,) + (pseudoSize), requires_grad=True)
        self.pseudoLR = pseudoLR
        self.pseudoOptimizer = torch.optim.SGD([self.pseudoData], lr=self.pseudoLR)
        
    def save(self, epoch, student, teacher, optimizer, subDirectory=None):
        """
        Save checkpoint of models.
        """
        self.checkpointDir = os.path.join('checkpoint', '' if subDirectory is None else subDirectory, self.currentTime)
        os.makedirs(self.checkpointDir, exist_ok=True)
        torch.save({'epoch': epoch,
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'pseudoData': self.pseudoData,
                    'studentOptimizer': optimizer.state_dict()},
                   os.path.join(self.checkpointDir, 'checkpoint.pt'))
        
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
                    optimizer.load_state_dict(state['studentOptimizer'])

                self.pseudoData = state['pseudoData']
                
        return startEpoch
    
    def log_images(self, epoch, n=4):
        """
        Log pseudo samples to tensorboard.
        
        :param epoch: int, current training epoch
        :param pseudoData: tensor, current pseudo samples
        :param n: int, amount of images to save
        """
        self.logger.add_images('Pseudo data', self.pseudoData[0:n], global_step=epoch, dataformats='NCHW')
        self.logger.flush()
        
    def train_step(self, student, teacher, dataloader, optimizer, objective, distillObjective):
        for _ in range(self.pseudoIters): # Pseudo steps
            pseudoMetrics = self._pseudo_step(student=student,
                                              teacher=teacher,
                                              objective=objective)
        for _ in range(self.studentIters): # Student steps
            studentMetrics = self._student_step(student=student,
                                                teacher=teacher,
                                                objective=objective,
                                                optimizer=optimizer)
        # Return pseudo and student metrics from last iteration in this epoch
        return {**pseudoMetrics, **studentMetrics}
        
    def _student_step(self, student, teacher, objective, optimizer):
        """
        Train student model to the teacher model.
        
        Initialize pseudoData as None before calling student_step.
        
        :return: (tensor, dict), tensor of samples and named metrics for logging.
        """
        student.train()
        teacher.eval()
        device = next(student.parameters()).device

        # Running metrics
        lossMeter = AverageMeter()
        
        # Calculate logits
        sLogits = student(self.pseudoData)
        tLogits = teacher(self.pseudoData).detach()
            
        # Calculate loss
        optimizer.zero_grad()
        batchLoss = objective(nn.functional.log_softmax(sLogits, dim=1), nn.functional.softmax(tLogits, dim=1))

        # Update student weights
        batchLoss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 5)
        optimizer.step()
            
        # Save metrics
        lossMeter.update(batchLoss.item(), n=self.pseudoData.size(0))
        return {'Loss/Student': lossMeter.avg}
    
    def _pseudo_step(self, student, teacher, objective):
        """
        Train pseudo data adversarially on the negative objective as discriminator.
        
        Note the objective should be provided in the usual manner, and it will automatically be reversed.
        
        :return: (tensor, dict), tensor of samples and named metrics for logging (negative pseudo loss).
        """
        #self.pseudoData = torch.rand((10,) + (3,32,32), requires_grad=True)
        #self.pseudoOptimizer = torch.optim.SGD([self.pseudoData], lr=self.pseudoLR)
        
        student.eval()
        teacher.eval()
        device = next(student.parameters()).device

        # Running metrics
        lossMeter = AverageMeter()

        # Rescale pseudo data
        if self.scaler is not None:
            rescaledData = self.scaler(self.pseudoData)
        else:
            rescaledData = self.pseudoData

        # Calculate logits
        sLogits = student(rescaledData)
        tLogits = teacher(rescaledData)

        # Calculate loss
        #self.pseudoOptimizer.zero_grad()
        batchLoss = -objective(nn.functional.log_softmax(sLogits, dim=1), nn.functional.softmax(tLogits, dim=1))

        # Update student weights
        batchLoss.backward()
        nn.utils.clip_grad_norm_(self.pseudoData, 5)
        self.pseudoOptimizer.step()
        
        # Save metrics
        lossMeter.update(batchLoss.item(), n=self.pseudoData.size(0))
        return {'Loss/Pseudo': -lossMeter.avg}
    
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
                'Valid/Metric': accMeter.avg}
    