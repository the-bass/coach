# Coach

A package for training your *PyTorch* network until the desired target was reached.

```py
coach = Coach(network=SimpleNet())
coach.train(
    target=coach.targets.loss_stagnation(epsilon=0.001),
    train_one_epoch=train_one_epoch,
    measure_performance=measure_performance,
    notes='LR=0.001',
    checkpoint_frequency=10,
    checkpoint=-1
)
```

*coach* supports *stateful modules* from the [torch_state_control](https://github.com/the-bass/torch_state_control) package. If you hand your network as an instance of `torch_state_control.nn.StatefulModule`, it will automatically save the parameters on each checkpoint along with the loss on train and dev set and other information.

Last tested with **Python 3.6.4 :: Anaconda, Inc.** and **PyTorch 0.4**.

## Usage

```py
from coach import Coach
import coach.targets
from networks.le_net_5 import LeNet5


network = LeNet5()
learning_rate = 0.001
coach = Coach(network=network)

def train_one_epoch():
    # ... train one epoch ...

    return epoch_loss

def measure_performance():
    train_set_performance = ...
    dev_set_performance = ...

    return train_set_performance, dev_set_performance

coach.train(
    target=coach.targets.loss_stagnation(epsilon=0.001),
    # target=coach.targets.time_elapsed(300), # in minutes
    # target=coach.targets.epoch_reached(500),
    train_one_epoch=train_one_epoch,
    measure_performance=measure_performance,
    checkpoint_frequency=4,
    checkpoint=-1, # Checkpoint to be loaded before training (only if you use a `StatefulModule`)
    notes=f"ADAM lr={learning_rate}", # Notes to be saved on every checkpoint (only if you use a `StatefulModule`)
    # record=False # Do not save parameters on each checkpoint (only if you use a `StatefulModule`)
)
```

The output will look something like

```sh
CHECKPOINT RESTORED | Checkpoint ID: 17, Success on TRAIN set: 0.0219144169, Success on DEV set: 0.0220695678 | 06:04:34, 16.05.2018
Epoch    2 | Loss: 0.021765273064375 | Time elapsed:  4.1 minutes                                                    
CHECKPOINT | Success on TRAIN set: 0.0212971214, Success on DEV set: 0.0214347895 | 07:35:57, 16.05.2018
Epoch    5 | Loss: 0.020889533683658 | Time elapsed: 12.6 minutes                                                    
CHECKPOINT | Success on TRAIN set: 0.0202702470, Success on DEV set: 0.0204370171 | 07:44:33, 16.05.2018
Epoch    7 | Loss: 0.020344411954284 | Time elapsed: 17.9 minutes                                                    
CHECKPOINT | Success on TRAIN set: 0.0196247473, Success on DEV set: 0.0197850894 | 07:49:40, 16.05.2018
Epoch    9 | Loss: 0.019553059712052 | Time elapsed: 22.8 minutes
```

## Installation

Clone this repository and run

```py
pip install .
```

inside the root directory to make the module available as `coach`.

## Development

*Unless noted otherwise, all commands are expected to be executed from the root directory of this repository.*

### Building the package for local development

To make the package available locally while making sure changes to the files are reflected immediately, run

```sh
pip install -e .
```

### Test suite

Run all tests using

```sh
python -m unittest discover tests
```
