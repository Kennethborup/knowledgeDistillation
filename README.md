![Hinton Knowledge Distillation](/hintonKD.png)

# Knowledge Distillation in PyTorch
Simple PyTorch implementation of (Hinton) Knowledge Distillation and BaseDistiller class to easily extend to other distillation procedures as well. Knowledge distillation in the sense of Hinton et al. (2015) seek to transfer *knowledge* from a large pretrained model, *teacher*, to a smaller untrained model, *student*. If done correctly, one can obtain performance improvements over student models trained from scratch, and more recent adaptions of the knowledge distillation scheme has examples of students outperforming the teacher. More recent work has introduced different distillation losses, looked at different information to transfer from the teacher, and the size of the student amongst others.

## Install requirements
To install the needed requirements in a new conda environment (HKD) use

```bash
conda env create -f environment.yml
```

## Example
Using the `HintonDistiller` is straight forward. Provide the usual elements; optimizer, objectives, models etc. and initiate the distiller with a weighting, `alpha`, between the distillation and objective function as well as the layers used for activation matching between student and teacher.

```python
import torch
import torch.nn as nn
from distillation.hintonDistiller import HintonDistiller
from distillation.utils import MLP, PseudoDataset

# Initialize random models and distiller
student = MLP(100, 10, 256)
teacher = MLP(100, 10, 256)
distiller = HintonDistiller(alpha=0.5, studentLayer=-2, teacherLayer=-2)

# Initialize objectives and optimizer
objective = nn.CrossEntropyLoss()
distillObjective = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

# Pseudo dataset and dataloader 
trainloader = torch.utils.data.DataLoader(
    PseudoDataset(size=(100)),
    batch_size=512,
    shuffle=True)

# Load state if checkpoint is provided
checkpoint = None
startEpoch = distiller.load_state(checkpoint, student, teacher, optimizer)
epochs = 15

# Construct tensorboard logger
distiller.init_tensorboard_logger()

for epoch in range(startEpoch, epochs+1):
        # Training step for one full epoch
        trainMetrics = distiller.train_step(student=student,
                                            teacher=teacher,
                                            dataloader=trainloader,
                                            optimizer=optimizer,
                                            objective=objective,
                                            distillObjective=distillObjective)
        
        # Validation step for one full epoch
        validMetrics = distiller.validate(student=student,
                                          dataloader=trainloader,
                                          objective=objective)
        metrics = {**trainMetrics, **validMetrics}
        
        # Log to tensorbard
        distiller.log(epoch, metrics)

        # Save model
        distiller.save(epoch, student, teacher, optimizer)
        
        # Print epoch performance
        distiller.print_epoch(epoch, epochs, metrics)
```

To continue a previous run, add the path to the checkpoint and adjust the `epochs` to the total training length. If only some elements from a previous run should be loaded, set the remaining arguments to `None` in the `.load_state()` call.

## Change type of knowledge distillation
In order to change the type of knowledge distillation, you merely need to change the type of `distiller`. Note, the following types of knowledge distillation is currently implemented:
 - Hinton Knowledge Distillation (Hinton et al. (2015))
 - Attention Knowledge Distillation (Zagoruyko and Komodakis (2016))
 - Data Free Knowledge Distillation or Zero-Shot Knowledge Distillation (Micaelli and Storkey (2019))
 
 For Attention Knowledge Distillation on the first and third layer change to the following.

```python
from distillation.attentionDistiller import AttentionDistiller
distiller = AttentionDistiller(alpha=0.5, studentLayer=[1, 3], teacherLayer=[1, 3])
```

Using the `DataFreeDistller` for Data Free Adversarial Knowledge Distillation (aka Zero-Shot Knowledge Distillation) is slightly more involved than e.g. `HintonDistiller` or `AttentionDistiller`. See the below example for usage of the `DataFreeDistiller`.

```python
import torch
import torch.nn as nn
from distillation.datafreeDistiller import DataFreeDistiller
from distillation.utils import PseudoDataset, CNN, Generator

# Initialize random models and distiller
imgSize = (3, 32, 32)
noiseDim = 100
student = CNN(imgSize, 64)
teacher = CNN(imgSize, 64)
distiller = DataFreeDistiller(generatorIters=3,
                              studentIters=2,
                              generator=Generator(noiseDim, imgSize),
                              generatorLR=1e-3,
                              batchSize=64,
                              noiseDim=noiseDim,
                              resampleRatio=1)

# Initialize objectives and optimizer
objective = nn.KLDivLoss(reduction='batchmean')
validObjective = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

# Pseudo dataset and dataloader 
validloader = torch.utils.data.DataLoader(
    PseudoDataset(size=imgSize),
    batch_size=512,
    shuffle=True)

# Load state if checkpoint is provided
checkpoint = None
startEpoch = distiller.load_state(checkpoint, student, teacher, optimizer)
epochs = 15

# Construct tensorboard logger
distiller.init_tensorboard_logger()

for epoch in range(startEpoch, epochs+1):
        # Training step for one full epoch
        trainMetrics = distiller.train_step(student=student,
                                            teacher=teacher,
                                            dataloader=None,
                                            optimizer=optimizer,
                                            objective=objective,
                                            distillObjective=None)
        
        # Validation step for one full epoch
        validMetrics = distiller.validate(student=student,
                                          dataloader=validloader,
                                          objective=validObjective)
        metrics = {**trainMetrics, **validMetrics}
        
        # Log to tensorbard
        distiller.log(epoch, metrics)

        # Save model
        distiller.save(epoch, student, teacher, optimizer)
        
        # Print epoch performance
        distiller.print_epoch(epoch, epochs, metrics)
```

## Citation
Remember to cite the original papers:

##### Hinton et. al (2015)
```bibtex
@misc{hinton2015distilling,
    title = {{Distilling the Knowledge in a Neural Network}},
    author = {Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
    year = {2015},
    eprint = {1503.02531},
    archivePrefix = {arXiv},
    primaryClass = {stat.ML}
}
```

##### Zagoruyko and Komodakis (2016)
```bibtex
@misc{zagoruyko2016paying,
    title = {{Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer}},
    author = {Zagoruyko, Sergey and Komodakis, Nikos},
    year = {2016},
    eprint = {1612.03928},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
}
```

##### Micaelli and Storkey (2019)
```bibtex
@misc{micaelli2019zeroshot,
    title = {{Zero-shot Knowledge Transfer via Adversarial Belief Matching}},
    author = {Micaelli, Paul and Storkey, Amos},
    year = {2019},
    eprint = {1905.09768},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
}

```