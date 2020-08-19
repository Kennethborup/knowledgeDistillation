import torch
import torch.nn as nn
from distillation.utils import Accuracy, AverageMeter, Hook, Generator
from distillation.baseDistiller import BaseDistiller
import random
import os

class DataFreeDistiller(BaseDistiller):
    def __init__(self, generatorIters, studentIters, generatorLR, batchSize=64, noiseDim=100, imgSize=(3,32,32), resampleRatio=1):
        super(DataFreeDistiller, self).__init__()
        # Training scheme
        self.studentIters = studentIters
        self.generatorIters = generatorIters
        self.epochIters = self.studentIters + self.generatorIters
        
        # Generator params
        self.batchSize = batchSize
        self.noiseDim = noiseDim
        self.pseudoData = None
        
        # Generator and optimizer XXX
        self.generator = Generator(self.noiseDim, imgSize)
        self.generatorOptimizer = torch.optim.SGD(self.generator.parameters(), lr=generatorLR)
        
        # Sampling scheme
        resampleRatio = resampleRatio if type(resampleRatio) in [list, tuple] else [resampleRatio]
        self.studentRR = resampleRatio[0]
        self.generatorRR = resampleRatio[-1]
        
       
    
    def _resample(self, pseudoData, generator, resampleRatio):
        device = next(generator.parameters()).device
        
        # If pseudoData is none (i.e. not initialized yet) generate full batch
        if pseudoData is None:
            pseudoData = generator(torch.randn((self.batchSize, self.noiseDim)).to(device))
        
        if resampleRatio > 0:
            # Indices to resample
            nResample = int(self.batchSize * resampleRatio)
            idx = torch.tensor(random.sample(range(self.batchSize), nResample))
            
            # Resample
            pseudoData[idx] = generator(torch.randn((nResample, self.noiseDim)).to(device))
        return pseudoData
    
    def save(self, epoch, student, teacher, optimizer, subDirectory=None):
        """
        Save checkpoint of models.
        """
        self.checkpointDir = os.path.join('checkpoint', '' if subDirectory is None else subDirectory, self.currentTime)
        os.makedirs(self.checkpointDir, exist_ok=True)
        torch.save({'epoch': epoch,
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'generator': self.generator.state_dict(),
                    'studentOptimizer': optimizer.state_dict(),
                    'generatorOptimizer': self.generatorOptimizer.state_dict()},
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

                self.generator.load_state_dict(state['generator'])
                self.generatorOptimizer.load_state_dict(state['generatorOptimizer'])
                
        return startEpoch
    
    def log_images(self, epoch, pseudoData, n=4):
        """
        Log pseudo samples to tensorboard.
        
        :param epoch: int, current training epoch
        :param pseudoData: tensor, current pseudo samples
        :param n: int, amount of images to save
        """
        self.logger.add_images('Pseudo data', pseudoData[0:n], global_step=epoch, dataformats='NCHW')
        self.logger.flush()
        
    def train_step(self, student, teacher, dataloader, optimizer, objective, distillObjective):
        for _ in range(self.generatorIters): # Generator steps
            self.pseudoData, generatorMetrics = self._generator_step(pseudoData=self.pseudoData,
                                                                     student=student,
                                                                     teacher=teacher,
                                                                     generator=self.generator,
                                                                     objective=objective,
                                                                     optimizer=self.generatorOptimizer)
        for _ in range(self.studentIters): # Student steps
            self.pseudoData, studentMetrics = self._student_step(pseudoData=self.pseudoData,
                                                                 student=student,
                                                                 teacher=teacher,
                                                                 generator=self.generator,
                                                                 objective=objective,
                                                                 optimizer=optimizer)
        # Return generator and student metrics from last iteration in this epoch
        return {**generatorMetrics, **studentMetrics}
        
    def _student_step(self, pseudoData, student, teacher, generator, objective, optimizer):
        """
        Train student model to the teacher model.
        
        Initialize pseudoData as None before calling student_step.
        
        :return: (tensor, dict), tensor of samples and named metrics for logging.
        """
        student.train()
        teacher.eval()
        generator.eval()
        device = next(student.parameters()).device

        # Running metrics
        lossMeter = AverageMeter()
        
        # Generate pseudo data
        pseudoData = self._resample(pseudoData, generator, self.studentRR).detach()
        
        # Calculate logits
        sLogits = student(pseudoData)
        tLogits = teacher(pseudoData).detach()
            
        # Calculate loss
        optimizer.zero_grad()
        batchLoss = objective(nn.functional.log_softmax(sLogits, dim=1), nn.functional.softmax(tLogits, dim=1))

        # Update student weights
        batchLoss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 5)
        optimizer.step()
            
        # Save metrics
        lossMeter.update(batchLoss.item(), n=pseudoData.size(0))
        return pseudoData, {'Loss/Student': lossMeter.avg}
    
    def _generator_step(self, pseudoData, student, teacher, generator, objective, optimizer):
        """
        Train generator adversarially on the negative objective as discriminator.
        
        Note the objective should be provided in the usual manner, and it will automatically be reversed.
        
        :return: (tensor, dict), tensor of samples and named metrics for logging (negative generator loss).
        """
        student.eval()
        teacher.eval()
        generator.train()
        device = next(student.parameters()).device
        
        # Running metrics
        lossMeter = AverageMeter()
        
        # Generate pseudo data
        pseudoData = self._resample(pseudoData, generator, self.generatorRR)
        
        # Calculate logits
        sLogits = student(pseudoData)
        tLogits = teacher(pseudoData)
            
        # Calculate loss
        optimizer.zero_grad()
        batchLoss = -objective(nn.functional.log_softmax(sLogits, dim=1), nn.functional.softmax(tLogits, dim=1))

        # Update student weights
        batchLoss.backward()
        #nn.utils.clip_grad_norm_(generator.parameters(), 5)
        optimizer.step()
            
        # Save metrics
        lossMeter.update(batchLoss.item(), n=pseudoData.size(0))
        return pseudoData.detach(), {'Loss/Generator': -lossMeter.avg}
    
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
    