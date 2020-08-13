![Hinton Knowledge Distillation](/hintonKD.png)

# Knowledge Distillation in PyTorch
Simple PyTorch implementation of (Hinton) Knowledge Distillation and BaseDistiller class to easily extend to other distillation procedures as well.

## Install requirements
To install the needed requirements in a new conda environment (HKD) use

```bash
conda env create -f environment.yml
```

## Example
Using the HintonDistiller is straight forward. Provide the usual elements; optimizer, objectives, models etc. and initiate the distiller with a weighting, `alpha`, between the distillation and objective function as well as the layers used for activation matching between student and teacher.

```python
import torch
import torch.nn as nn
from distillation.hintonDistiller import HintonDistiller
from distillation.utils import MLP, PseudoDataset

# Initialize random models and distiller
student = MLP(100, 10, 256)
teacher = MLP(100, 10, 256)
distiller = HintonDistiller(alpha=0.5, studentLayer=-1, teacherLayer=-1)

# Initialize objectives and optimizer
objective = nn.KLDivLoss(reduction='batchmean')
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
        trainMetrics = distiller.train_step(epoch=epoch,
                                            student=student,
                                            teacher=teacher,
                                            dataloader=trainloader,
                                            objective=objective,
                                            distillObjective=distillObjective,
                                            optimizer=optimizer)
        
        # Log to tensorbard
        distiller.log(epoch, trainMetrics)

        # Save model
        distiller.save(epoch, student, teacher, optimizer)
        
        # Print epoch performance
        distiller.print_epoch(epoch, epochs, trainMetrics)
```

To continue a previous run, add the path to the checkpoint and adjust the `epochs` to the total training length. If only some elements from a previous run should be loaded, set the remaining arguments to None in the `.load_state()` call.


## Citation
Remember to cite the original paper by Hinton et al. (2015).

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